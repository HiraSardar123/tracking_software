"""Derive a documented, reproducible set of benchmark intervals from the
reference tracks.

The original 17 AlS28T4 intervals were not distributed with the data, so this
script defines them by an explicit rule instead of guessing. The rule:

  1. Load the reference droplet tracks (class_id=3) and take each track's
     [first_frame, last_frame] span. Each track is one droplet detachment
     event, so events -- not arbitrary time slices -- anchor the intervals.
  2. Sort events by first_frame and select N of them at evenly spaced ranks
     (rank = round(i * (n_events - 1) / (N - 1))). Even spacing over event
     rank, not over raw frame index, keeps sampling proportional to droplet
     activity rather than over-weighting idle stretches.
  3. Centre a window of --length frames on each selected event's midpoint and
     clip it to the video bounds.
  4. Drop windows that overlap an already accepted window, walking to the next
     unused event rank, so the N intervals stay disjoint.

Default length is 270 frames: the reference Stage-1 time (~21.5 s) divided by
this pipeline's measured detection cost (~0.079 s/frame) implies the original
windows were about that size.

    python -m tools.make_intervals --gt ../AlS28T4/tracks_clean.parquet \
        --n 17 --length 270 --out configs/intervals_AlS28T4.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def build(gt_path, n, length, class_id, n_frames):
    df = pd.read_parquet(gt_path)
    if "class_id" in df.columns:
        df = df[df["class_id"] == class_id]

    ev = (df.groupby("global_id")["abs_frame"]
            .agg(["min", "max"]).reset_index()
            .sort_values("min").reset_index(drop=True))
    n_ev = len(ev)
    if n_ev < n:
        raise SystemExit(f"only {n_ev} reference events; cannot pick {n} intervals")

    half = length // 2
    lo_bound, hi_bound = 0, n_frames - 1

    chosen, used = [], set()
    for i in range(n):
        rank = round(i * (n_ev - 1) / (n - 1))
        # Walk forward to the next unused event whose window stays disjoint.
        for cand in list(range(rank, n_ev)) + list(range(rank - 1, -1, -1)):
            if cand in used:
                continue
            row = ev.iloc[cand]
            mid = int((int(row["min"]) + int(row["max"])) // 2)
            st = max(lo_bound, mid - half)
            en = min(hi_bound, st + length - 1)
            st = max(lo_bound, en - length + 1)
            if any(not (en < c["start"] or st > c["end"]) for c in chosen):
                continue
            used.add(cand)
            chosen.append({
                "name": f"iv{len(chosen):02d}",
                "start": int(st),
                "end": int(en),
                "anchor_global_id": int(row["global_id"]),
                "anchor_span": [int(row["min"]), int(row["max"])],
                # Categories are assigned by the benchmark from measured
                # ID-consistency at skip=1; see tools/benchmark_eval.py.
                "category": None,
            })
            break

    chosen.sort(key=lambda c: c["start"])
    for i, c in enumerate(chosen):
        c["name"] = f"iv{i:02d}"
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--n", type=int, default=17)
    ap.add_argument("--length", type=int, default=270)
    ap.add_argument("--class-id", type=int, default=3)
    ap.add_argument("--n-frames", type=int, default=14999)
    ap.add_argument("--out", default="configs/intervals_AlS28T4.json")
    args = ap.parse_args()

    iv = build(args.gt, args.n, args.length, args.class_id, args.n_frames)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(iv, f, indent=2)

    print(f"[intervals] wrote {len(iv)} intervals -> {args.out}")
    for c in iv:
        print(f"  {c['name']}  frames {c['start']:>6}-{c['end']:>6}  "
              f"(anchor track {c['anchor_global_id']}, span {c['anchor_span']})")


if __name__ == "__main__":
    main()
