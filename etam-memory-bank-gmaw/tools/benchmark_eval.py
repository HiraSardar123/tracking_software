"""Interval x memory-update-skip benchmark against the reference tracks.

Runs Stages 1-3 over a set of frame intervals for each skip setting, times each
stage, and scores the predicted droplet masks/tracks against the reference
parquet. Emits three tables: overall, per-skip ablation, per-category.

    python -m tools.benchmark_eval \
        --gt ../AlS28T4/tracks_clean.parquet \
        --intervals configs/intervals_AlS28T4.json \
        --skips 1,3,5 --out outputs/benchmark

Metrics per (interval, skip)
---------------------------
runtime      stage1 + stage2 + stage3 seconds (model load/init excluded, so the
             number is comparable across intervals)
IoU / Dice   per-frame agreement between the union of predicted droplet masks
             and the union of reference droplet masks, averaged over the
             interval. Frames where both are empty score 1.0.
track cont.  per predicted track, observed frames / (last - first + 1),
             averaged over tracks. 1.0 means no gaps.
count error  signed (predicted unique tracks - reference unique tracks).


Category assignment
-------------------
The original human `incorrect_id_assignment` labels were not distributed with
the data. This script substitutes a measured, reproducible proxy computed at
skip=1: a reference track and a predicted track are associated when their
overlap reaches --assoc-frac of the smaller mask-pixel total. An interval is
labelled `incorrect_id_assignment` when any reference track associates with 2+
predicted tracks (an ID split) or any predicted track associates with 2+
reference tracks (an ID merge); otherwise `uncategorized`. The label is decided
once at skip=1 and reused for that interval's other skip settings so the
category grouping stays fixed across the ablation.
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import EfficientTAMTracker
from src.pipeline import (
    stage1_detect_and_segment,
    stage2_temporal_filter_forward_iou,
    stage3_track_and_discover,
)
from src.utils import load_frame_names


# ----------------------------- ground truth -----------------------------
def load_gt(gt_path, W, H, class_id=3):
    """-> {abs_frame: {global_id: sorted linear pixel indices}}"""
    df = pd.read_parquet(gt_path)
    if "class_id" in df.columns:
        df = df[df["class_id"] == class_id]
    out = defaultdict(dict)
    limit = H * W
    for row in df.itertuples(index=False):
        idx = np.asarray(row.mask_px, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < limit)]
        out[int(row.abs_frame)][int(row.global_id)] = np.unique(idx)
    return dict(out)


def gt_union(gt_frame, H, W):
    m = np.zeros(H * W, dtype=bool)
    for idx in gt_frame.values():
        m[idx] = True
    return m.reshape(H, W)


# ----------------------------- metrics -----------------------------
def mask_scores(pred_by_frame, gt, frames, H, W):
    ious, dices = [], []
    empty = np.zeros((H, W), dtype=bool)
    for f in frames:
        p = pred_by_frame.get(f, empty)
        gf = gt.get(f)
        g = gt_union(gf, H, W) if gf else empty
        inter = int(np.logical_and(p, g).sum())
        union = int(np.logical_or(p, g).sum())
        if union == 0:
            ious.append(1.0)
            dices.append(1.0)
            continue
        psum, gsum = int(p.sum()), int(g.sum())
        ious.append(inter / union)
        dices.append((2 * inter) / (psum + gsum) if (psum + gsum) else 0.0)
    return (float(np.mean(ious)) if ious else 0.0,
            float(np.mean(dices)) if dices else 0.0)


def track_continuity(T):
    vals = []
    for per_frame in T.values():
        fs = sorted(per_frame.keys())
        if not fs:
            continue
        span = fs[-1] - fs[0] + 1
        vals.append(len(fs) / span if span > 0 else 1.0)
    return float(np.mean(vals)) if vals else 1.0


ASSOC_FRACS = (0.10, 0.30, 0.50, 0.70)


def id_consistency(T, gt, start, W, fracs=ASSOC_FRACS):
    """Detect ID splits/merges between predicted and reference tracks.

    Overlaps are accumulated once and thresholded at several association
    fractions, so the category rule can be re-derived later without re-running
    the pipeline. Returns {frac: (splits, merges)}.
    """
    ov = defaultdict(int)
    tot_pred, tot_gt = defaultdict(int), defaultdict(int)

    for oid, per_frame in T.items():
        for f_loc, (ys, xs) in per_frame.items():
            f_abs = start + int(f_loc)
            p = np.unique(np.asarray(ys, np.int64) * W + np.asarray(xs, np.int64))
            tot_pred[oid] += p.size
            for gid, g in gt.get(f_abs, {}).items():
                n = np.intersect1d(p, g, assume_unique=True).size
                if n:
                    ov[(oid, gid)] += n

    for f_abs, per_id in gt.items():
        if start <= f_abs:
            for gid, g in per_id.items():
                tot_gt[gid] += g.size

    out = {}
    for frac in fracs:
        preds_of_gt, gts_of_pred = defaultdict(set), defaultdict(set)
        for (oid, gid), n in ov.items():
            if n >= frac * min(tot_pred[oid], tot_gt[gid]):
                preds_of_gt[gid].add(oid)
                gts_of_pred[oid].add(gid)
        splits = sum(1 for v in preds_of_gt.values() if len(v) > 1)
        merges = sum(1 for v in gts_of_pred.values() if len(v) > 1)
        out[frac] = (splits, merges)
    return out


# ----------------------------- one evaluation -----------------------------
def make_view(video_dir, names, st, en, dest):
    """EfficientTAM loads a whole folder, so expose only this interval."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    for f in range(st, en + 1):
        src, dst = Path(video_dir) / names[f], dest / names[f]
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    return dest


def run_one(cfg, video_dir, names, iv, skip, gt, W, H, tmp_root):
    st, en = iv["start"], iv["end"]
    view = make_view(video_dir, names, st, en, Path(tmp_root) / f"{iv['name']}_s{skip}")
    view_names = load_frame_names(str(view))

    cfg = json.loads(json.dumps(cfg))
    cfg["stage3"]["memory_update_skip"] = int(skip)

    t_wall = time.time()
    etam = EfficientTAMTracker(
        cfg_path=cfg["efficienttam"]["cfg"],
        ckpt_path=cfg["efficienttam"]["ckpt"],
        device=cfg["run"].get("device", "auto"),
        compile_image_encoder=False,
    )
    etam.init(str(view))

    track_cls = int(cfg["yolo"]["classes"]["track"][0])
    seg_classes = sorted({track_cls, *[int(c) for c in cfg["yolo"]["classes"].get("segment_only", [])]})

    boxes_cls, indep_cls, t1 = stage1_detect_and_segment(
        cfg, str(view), view_names, H, W, etam=etam, offset=0, classes_to_segment=seg_classes
    )
    n = len(view_names)
    boxes = {i: boxes_cls.get(i, {}).get(track_cls, np.zeros((0, 4), np.float32)) for i in range(n)}
    indep = {}
    for i in range(n):
        e = indep_cls.get(i, {}).get(track_cls, {"mask_ids": {}, "boxes": np.zeros((0, 4), np.float32)})
        indep[i] = {"mask_ids": dict(e["mask_ids"]), "boxes": e["boxes"]}

    boxes_f, masks_f, t2 = stage2_temporal_filter_forward_iou(cfg, boxes, indep, H, W)
    T, t3 = stage3_track_and_discover(cfg, etam, view_names, boxes_f, masks_f, H, W, offset=0)

    runtime = t1 + t2 + t3          # matches the paper's stage-sum convention
    wall = time.time() - t_wall

    pred_by_frame = {}
    for per_frame in T.values():
        for f_loc, (ys, xs) in per_frame.items():
            f_abs = st + int(f_loc)
            m = pred_by_frame.get(f_abs)
            if m is None:
                m = np.zeros((H, W), dtype=bool)
                pred_by_frame[f_abs] = m
            m[ys, xs] = True

    frames = list(range(st, en + 1))
    iou, dice = mask_scores(pred_by_frame, gt, frames, H, W)
    cont = track_continuity(T)

    gt_in = set()
    for f in frames:
        gt_in |= set(gt.get(f, {}).keys())
    gt_count, pred_count = len(gt_in), len(T)

    gt_local = {f: gt[f] for f in frames if f in gt}
    idc = id_consistency(T, gt_local, st, W)

    shutil.rmtree(view, ignore_errors=True)
    del etam

    return {
        "interval": iv["name"], "start": st, "end": en, "skip": skip,
        "runtime_s": runtime, "wall_s": wall,
        "stage1_s": t1, "stage2_s": t2, "stage3_s": t3,
        "iou": iou, "dice": dice, "track_continuity": cont,
        "pred_count": pred_count, "gt_count": gt_count,
        "count_error": pred_count - gt_count,
        **{f"id_splits@{f:g}": idc[f][0] for f in ASSOC_FRACS},
        **{f"id_merges@{f:g}": idc[f][1] for f in ASSOC_FRACS},
    }


# ----------------------------- tables -----------------------------
def emit_tables(df, out_dir):
    L = []
    L.append("TABLE I: Overall benchmark statistics across the evaluated AlS28T4 "
             "intervals and skip settings.\n")
    L.append(f"{'Metric':<30}{'Value':>12}")
    for k, v in [
        ("Videos evaluated", 1),
        ("Unique intervals", df["interval"].nunique()),
        ("Interval-skip evaluations", len(df)),
        ("Mean total runtime (s)", df["runtime_s"].mean()),
        ("Mean IoU", df["iou"].mean()),
        ("Mean Dice", df["dice"].mean()),
        ("Mean track continuity", df["track_continuity"].mean()),
        ("Mean count error", df["count_error"].mean()),
    ]:
        L.append(f"{k:<30}{(f'{v:.4f}' if isinstance(v, float) else v):>12}")

    L.append("\n\nTABLE II: Ablation results for the memory-update skip parameter.\n")
    L.append(f"{'Skip':<6}{'Runtime(s)':>12}{'Stage1(s)':>12}{'Stage2(s)':>12}"
             f"{'Stage3(s)':>12}{'IoU':>10}{'Dice':>10}{'Track':>10}{'Counterror':>12}")
    for skip, g in df.groupby("skip"):
        L.append(f"{skip:<6}{g['runtime_s'].mean():>12.4f}{g['stage1_s'].mean():>12.4f}"
                 f"{g['stage2_s'].mean():>12.4f}{g['stage3_s'].mean():>12.4f}"
                 f"{g['iou'].mean():>10.4f}{g['dice'].mean():>10.4f}"
                 f"{g['track_continuity'].mean():>10.4f}{g['count_error'].mean():>12.4f}")

    L.append("\n\nTABLE III: Category-wise benchmark summary.\n")
    L.append(f"{'Category':<26}{'Meanruntime(s)':>16}{'MeanIoU':>10}{'MeanDice':>10}"
             f"{'Trackcont':>12}{'Counterror':>12}{'Intervals':>11}")
    for cat, g in df.groupby("category"):
        L.append(f"{cat:<26}{g['runtime_s'].mean():>16.4f}{g['iou'].mean():>10.4f}"
                 f"{g['dice'].mean():>10.4f}{g['track_continuity'].mean():>12.4f}"
                 f"{g['count_error'].mean():>12.4f}{g['interval'].nunique():>11}")

    text = "\n".join(L)
    (Path(out_dir) / "benchmark_tables.txt").write_text(text, encoding="utf-8")
    print("\n" + text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/pipeline.yaml")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--intervals", required=True)
    ap.add_argument("--skips", default="1,3,5")
    ap.add_argument("--gt-class-id", type=int, default=3)
    ap.add_argument("--assoc-frac", type=float, default=0.10)
    ap.add_argument("--out", default="outputs/benchmark")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    video_dir = Path(cfg["data"]["video_dir"])
    names = load_frame_names(str(video_dir))
    from PIL import Image
    W, H = Image.open(video_dir / names[0]).size

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = out_dir / "_views"

    intervals = json.load(open(args.intervals))
    skips = sorted({int(s) for s in args.skips.split(",") if s.strip()})
    print(f"[bench] {len(intervals)} intervals x {len(skips)} skips "
          f"= {len(intervals)*len(skips)} interval-skip evaluations")

    gt = load_gt(args.gt, W, H, class_id=args.gt_class_id)

    rows = []
    for iv in intervals:
        for sk in skips:
            r = run_one(cfg, video_dir, names, iv, sk, gt, W, H, tmp_root)
            rows.append(r)
            print(f"  {r['interval']:<7} skip={sk}  rt={r['runtime_s']:7.2f}s "
                  f"(s1={r['stage1_s']:6.2f} s2={r['stage2_s']:5.2f} s3={r['stage3_s']:7.2f})  "
                  f"IoU={r['iou']:.4f} Dice={r['dice']:.4f} cont={r['track_continuity']:.4f} "
                  f"cerr={r['count_error']:+d} "
                  f"split={r[f'id_splits@{args.assoc_frac:g}']} "
                  f"merge={r[f'id_merges@{args.assoc_frac:g}']}")
            pd.DataFrame(rows).to_csv(out_dir / "benchmark_raw.csv", index=False)

    df = pd.DataFrame(rows)

    # Category decided once at the smallest skip, then applied to every skip.
    base = df[df["skip"] == skips[0]].set_index("interval")
    sc, mc = f"id_splits@{args.assoc_frac:g}", f"id_merges@{args.assoc_frac:g}"
    cat = {iv: ("incorrect_id_assignment"
                if (base.loc[iv, sc] > 0 or base.loc[iv, mc] > 0)
                else "uncategorized")
           for iv in base.index}
    df["category"] = df["interval"].map(cat)
    print(f"[bench] categories at assoc_frac={args.assoc_frac:g}: "
          f"{sum(v == 'incorrect_id_assignment' for v in cat.values())} "
          f"incorrect_id_assignment / {len(cat)} intervals")
    for f in ASSOC_FRACS:
        b = df[df["skip"] == skips[0]].set_index("interval")
        n = sum(1 for iv in b.index
                if b.loc[iv, f"id_splits@{f:g}"] > 0 or b.loc[iv, f"id_merges@{f:g}"] > 0)
        print(f"          (assoc_frac={f:g} would give {n}/{len(b)})")

    df.to_csv(out_dir / "benchmark_raw.csv", index=False)
    with open(out_dir / "interval_categories.json", "w") as f:
        json.dump(cat, f, indent=2)

    shutil.rmtree(tmp_root, ignore_errors=True)
    emit_tables(df, out_dir)
    print(f"[bench] raw  -> {out_dir/'benchmark_raw.csv'}")
    print(f"[bench] cats -> {out_dir/'interval_categories.json'}")


if __name__ == "__main__":
    main()
