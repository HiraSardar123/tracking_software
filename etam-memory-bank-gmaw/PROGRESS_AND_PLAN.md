# HS-WeldCount — Progress, Results, and the New-Filter Plan

Branch: `new-filter` · Hardware: RTX 4060 Ti (8 GB), 64 GB RAM · Date: 2026-08-18

---

## 1. What has been done

### 1.1 Reproduction of the AlS28T4 run

The pipeline was brought up from scratch on Windows and run end-to-end over all
14,999 frames. Five things had to be fixed before it would run at all:

| # | Problem | Fix |
|---|---------|-----|
| 1 | `hydra-core`, `omegaconf`, `iopath` missing from `requirements.txt` | installed; they are hard imports of `build_efficienttam.py` |
| 2 | `init_state()` allocates `num_frames × 3 × 1024 × 1024` float32 = **188 GB** for the full video | chunking (`strategy: fixed_len`) + `offload_video_to_cpu/state=True` in `src/models.py` |
| 3 | Frames loaded into 8 GB VRAM | same offload change; VRAM use dropped to ~2.2 GB |
| 4 | `os.symlink` fails on Windows without admin (`WinError 1314`) | hardlink-first fallback chain in `src/run_chunks.py` |
| 5 | `--frame-start/--frame-end` do not limit what `init_state` loads | chunk views expose only the frames a chunk needs |

Run: 16 chunks × ~1000 frames, ~50 min wall time.

### 1.2 Result vs the reference package

The `counts.json` value of **36 is not the track count** — it does not correspond
to the reference track table and should be ignored. The real comparison is
against `AlS28T4/tracks_clean.parquet`:

| | Reference | Ours | Delta |
|---|---|---|---|
| Unique droplet tracks (raw) | 121 | 133 | +10% |
| **At min_len ≥ 21 frames** | **121** | **122** | **+0.8%** |
| Mask rows | 13,452 | 12,501 | −7% |

The reference's shortest track is exactly 21 frames, i.e. their post-processing
dropped anything shorter; our config keeps `min_len_frames: 6`. Applying their
threshold gives 122 vs 121. **The reproduction is sound.**

Residual difference sources: (a) chunk-boundary fragmentation — 4 tracks begin
within 4 frames of a seam, and we used 16 seams vs their 4; (b) checkpoint
mismatch — our `best.pt` is 2-class (`consumable`, `droplet`), theirs was
5-class with `droplet = 3`.

### 1.3 Benchmark harness (new)

- `tools/make_intervals.py` — derives a documented, reproducible interval set.
- `tools/benchmark_eval.py` — runs Stages 1-3 per interval per skip, times each
  stage, scores against the reference, emits the three tables.
- `stage3.memory_update_skip` — the memory-update skip parameter (did not exist
  in the codebase; added to `src/pipeline.py`).
- `compile_image_encoder=False` for benchmarking, so torch.compile warmup is not
  charged to the first interval.

**Intervals.** The original 17 were not distributed with the data — no interval
list and no `incorrect_id_assignment` label exists anywhere in the repo or the
data package. We therefore define them by an explicit rule (see the docstring of
`tools/make_intervals.py`): anchored on reference droplet events, sampled at
evenly spaced event ranks, 270 frames wide. The width is derived, not guessed —
their Stage-1 time (~21.5 s) divided by our measured detection cost
(0.079 s/frame) implies ~270 frames, and our measured Stage 1 came out at 23.4 s.

**Categories.** Human labels were unavailable, so the category is a measured
proxy: an interval is `incorrect_id_assignment` when a reference track splits
across 2+ predicted tracks, or a predicted track absorbs 2+ reference tracks.
Split/merge counts are computed at four association thresholds
(0.10 / 0.30 / 0.50 / 0.70) in one pass, so the rule can be re-derived from the
CSV without re-running. At 0.10 the split is 9 / 8.

---

## 2. Benchmark results (baseline, legacy discovery gate)

17 intervals × 3 skips = 51 interval-skip evaluations.

### Table I — Overall

| Metric | Value |
|---|---|
| Videos evaluated | 1 |
| Unique intervals | 17 |
| Interval-skip evaluations | 51 |
| Mean total runtime (s) | 42.4089 |
| Mean IoU | 0.7919 |
| Mean Dice | 0.8338 |
| Mean track continuity | 0.9974 |
| Mean count error | 0.7059 |

### Table II — Memory-update skip ablation

| Skip | Runtime(s) | Stage1(s) | Stage2(s) | Stage3(s) | IoU | Dice | Track | Count error |
|---|---|---|---|---|---|---|---|---|
| 1 | 43.9428 | 23.4872 | 0.2385 | 20.2171 | 0.7788 | 0.8229 | 0.9984 | 1.1176 |
| 3 | 41.5803 | 23.2854 | 0.2190 | 18.0759 | 0.8056 | 0.8454 | 0.9969 | 0.5882 |
| 5 | 41.7038 | 23.4590 | 0.2259 | 18.0189 | 0.7913 | 0.8332 | 0.9968 | 0.4118 |

### Table III — Category-wise

| Category | Mean runtime(s) | Mean IoU | Mean Dice | Track cont. | Count error | Intervals |
|---|---|---|---|---|---|---|
| incorrect_id_assignment | 42.6249 | 0.7810 | 0.8226 | 0.9976 | 0.3704 | 9 |
| uncategorized | 42.1659 | 0.8041 | 0.8465 | 0.9971 | 1.0833 | 8 |

---

## 3. Differences from the reported tables, and why

| Metric | Report | Ours | Assessment |
|---|---|---|---|
| Track continuity | 0.9967 | 0.9974 | **matches** |
| IoU | 0.0504 | 0.7919 | ~15× apart |
| Dice | 0.0755 | 0.8338 | ~11× apart |
| Mean runtime | 83.29 s | 42.41 s | different hardware; compile disabled |
| Count error vs skip | rises 0.118 → 0.353 | **falls** 1.118 → 0.412 | opposite trend |
| `incorrect_id` runtime | slower (114.7 vs 73.6) | ~equal (42.6 vs 42.2) | proxy labels, not human labels |

**On IoU/Dice.** Track continuity matching to three decimals while IoU differs
by 15× is not a plausible signature of a different tracker — it is the signature
of a broken mask comparison. A pipeline that reproduces the reference to 122 vs
121 tracks cannot genuinely overlap its masks by 5%. Most likely cause in the
original evaluation: `mask_px` decoded with a mismatched width (the linear index
is `W*y + x`, W = 800), or compared against the wrong class id. **Recommend not
publishing 0.05 without re-deriving it.**

**On the count-error trend.** Ours is not noise, and the raw CSV explains it:

- At skip=1 the error is **+1.118 and never negative** — 14 of 17 intervals
  over-count, 0 under-count.
- At skip=5 it is +0.412, with over-count in 8 and under-count in 3.

So skip=1 is not "worse tracking". Skip=1 **runs discovery more often, and every
extra discovery pass creates duplicate tracks.** Raising the skip accidentally
suppresses duplicates by looking less often. Mean absolute error confirms it:
1.118 → 1.059 → 0.765.

---

## 4. Why we are lagging: the root cause

`_get_new_boxes_in_frame()` decided whether a detection was a *new* object with:

```python
if (union_tracked & m).any():   # one shared pixel suppresses
    continue
```

Two defects:

1. **Single frame.** It compares only against tracked masks at frame `f`. If a
   propagated mask lags the detector by one frame, the detection sees no overlap
   and seeds a duplicate track for an object already being tracked.
2. **Single pixel.** `.any()` is a binary all-or-nothing test. It is
   simultaneously too strict (one stray pixel of a neighbouring droplet
   suppresses a genuinely new one) and too weak (it never asks *how much* of the
   candidate is already covered).

CountVid — the paper this pipeline reimplements — does not do this. It uses an
**IoU threshold (0.5)** for masklet matching and **validates a new object over a
temporal window (w = 3) with backward tracking** before admitting it. Our Stage 2
implements the temporal window; **Stage 3 discovery does not.** That gap is the
over-count, and it is why `skip` looked like it "fixed" accuracy.

---

## 5. What Phase A changes (CountVid-style discovery gate) — IMPLEMENTED

All four changes live in `src/pipeline.py`, configured under `stage3.discovery`.
Defaults are legacy-equivalent, so the change is opt-in and A/B-able.

| Knob | Value | What it does | Paper basis |
|---|---|---|---|
| `window` | 2 | Compare candidate against tracked masks over frames `[f−2, f]`, not just `f` | fixes propagation lag |
| `overlap_frac` | 0.30 | Suppress when ≥30% of the candidate's own area is already tracked, instead of ≥1 pixel | CountVid IoU matching (0.5) |
| `persist_k` | 2 | Candidate must survive 2 consecutive discovery frames before it is seeded | CountVid temporal validation (w = 3) |
| `merge_iou` | 0.50 | After seeding, drop a new track whose masks IoU-match an existing track over shared frames | duplicate masklet removal |

New functions: `_confirm_persistent()`, `_merge_duplicate_tracks()`, `_box_iou()`;
`_get_new_boxes_in_frame()` gained `window` and `overlap_frac`.

**Expected effect.** Count error at skip=1 falls toward 0, and Stage 3 gets
*faster*, because each suppressed duplicate saves a full
`etam.reset()` + `seed_box()` + `propagate()` cycle. If the diagnosis is right,
the skip trend should also invert back to the expected direction (higher skip =
slightly worse recall), because skip will no longer be masking a seeding bug.

---

## 6. Remaining phases

**Phase B — event-triggered discovery (replaces fixed skip).** Run discovery
only where evidence of a new object exists, instead of on a blind stride.
Tier 1 (free): trigger when a YOLO box fails to IoU-match any alive track — the
signal already sits in `boxes_by_f`. Tier 2 (cheap): change-energy from a
downsampled frame difference or FFT high-band, i.e. the cosine-similarity idea
from SurgSAM2 applied to frame selection. At 13 kfps, detachments are abrupt and
idle stretches are near-identical, so this should give skip-5 runtime at skip-1
recall. Implement the trigger as an interface (`stride | det_mismatch |
change_energy`) so all three are benchmarkable.

**Phase C — memory-bank pruning (SurgSAM2 EFP).** ETAM keeps `num_maskmem: 7`
memory frames per object; at 13 kfps consecutive memories are nearly identical.
Score entries by cosine similarity of `maskmem_features` and keep only the most
informative. SurgSAM2 reports ~3× inference speedup. Cheap first test: set
`num_maskmem: 7 → 3` (one config line) and re-benchmark before touching
`memory_attention.py`.

**Phase D — Stage-1 frame filter.** Stage 1 is the largest fixed cost (23.4 s of
42.4 s, and skip-independent). The Phase-B change-energy signal can skip
YOLO + segmentation on near-identical frames and reuse previous detections. Large
win, but it perturbs counts directly — only after A and B validate.

**Also worth doing:** re-derive the reference IoU to confirm the 0.05 is a bug;
reduce chunk seams (`frames_per_part: 1000 → 2000`) to cut boundary
fragmentation; ask for the 5-class checkpoint.

---

## 7. How to A/B this

The existing `outputs/benchmark/` **is** the legacy baseline — it was produced
before the discovery gate existed. So:

```
.\hira-env\Scripts\python.exe -m tools.benchmark_eval --gt ..\AlS28T4\tracks_clean.parquet --intervals configs\intervals_AlS28T4.json --skips 1,3,5 --out outputs\benchmark_phaseA
```

Compare `outputs/benchmark/benchmark_tables.txt` (legacy) against
`outputs/benchmark_phaseA/benchmark_tables.txt`. The headline number to watch is
**count error at skip=1: 1.1176 → ?**

To restore legacy behaviour without editing code, set in `configs/pipeline.yaml`:
`window: 0`, `overlap_frac: 0.0`, `persist_k: 1`, `merge_iou: 0.0`.

---

## 8. Files

| Path | Role |
|---|---|
| `src/pipeline.py` | Stages 1-3; discovery gate + `memory_update_skip` |
| `src/models.py` | CPU offload; `compile_image_encoder` toggle |
| `src/run_chunks.py` | hardlink fallback for Windows |
| `tools/make_intervals.py` | interval definition (documented rule) |
| `tools/benchmark_eval.py` | benchmark + three tables |
| `configs/intervals_AlS28T4.json` | the 17 intervals |
| `outputs/benchmark/` | legacy baseline: tables, raw CSV, categories |
