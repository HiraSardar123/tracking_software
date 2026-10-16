# HS-WeldCount: Technical Architecture & Implementation Details

## System Overview

```
INPUT: Video frames (JPG/PNG folder)
  ↓
[STAGE 1] Per-Frame Detection + Segmentation
  ├─ YOLO v11x: Detect droplets + weld pool
  └─ EfficientTAM: Segment detected regions
  ↓ (Per-frame masks)
[STAGE 2] Temporal Deduplication
  └─ Forward IoU filter: Remove per-frame duplicates
  ↓ (Clean per-frame masks)
[STAGE 3] Long-Term Tracking
  └─ Global ID assignment + new object discovery
  ↓ (Track IDs across entire video)
[POST-PROCESS] (Optional, not in paper)
  ├─ Temporal smoothing
  ├─ Physics-based filters (falling droplets)
  └─ Domain-specific rules (stuck droplet removal)
  ↓
[OUTPUT] Tracks + Statistics
  ├─ Parquet tables
  ├─ JSON summaries
  ├─ PNG overlays
  └─ MP4 video
```

---

## Stage 1: Per-Frame Detection & Segmentation

### Data Flow

```
Frame 0:  img_0000.jpg
  ↓
YOLO Inference
  ├─ Bboxes for class 1 (droplets)
  ├─ Bboxes for class 0 (weld pool)
  └─ Confidence scores
  ↓
EfficientTAM Segmentation
  ├─ Input: frame + bboxes from YOLO
  ├─ State: None (fresh frame)
  ├─ Output: Binary masks (pixel-level)
  └─ Per-frame result:
      {
        "mask_ids": {0: (ys, xs), 1: (ys, xs), ...},
        "boxes": np.array([N, 4]),
        "class_labels": [0, 1, 0, ...]  # track vs segment_only
      }

Frame 1:  img_0001.jpg
  ↓
[Repeat independently]
```

### Implementation: `stage1_detect_and_segment()`

**File**: `src/pipeline.py`

**Pseudo-code**:

```python
def stage1_detect_and_segment(cfg, video_dir, frame_names, ...):
    # Initialize models
    yolo = YOLODetector(cfg["yolo"]["weights"], conf=0.15, ...)
    etam = EfficientTAMTracker(
        cfg_path="configs/efficienttam/efficienttam_s.yaml",
        ckpt_path="efficient_track_anything/checkpoints/efficienttam_s.pt"
    )
    
    per_frame_masks = {}  # {frame_idx: {mask_id: (ys, xs)}}
    
    for frame_idx, frame_name in enumerate(frame_names):
        img_path = os.path.join(video_dir, frame_name)
        
        # 1. YOLO detection
        boxes_by_class = yolo.detect_by_class(img_path)
        # boxes_by_class = {0: [box1, box2], 1: [box3, ...]}
        
        # 2. EfficientTAM segmentation
        masks = {}
        for class_id, boxes in boxes_by_class.items():
            if boxes.size == 0:
                continue
            
            # Segment all boxes for this class
            class_masks = etam.segment_boxes_at_frame(frame_idx, boxes)
            # class_masks: {local_id: (ys, xs), ...}
            
            masks.update(class_masks)
        
        per_frame_masks[frame_idx] = masks
    
    return per_frame_masks
```

### Key Details

**YOLO Input**: Frame path (string)
- Runs inference on single image
- Returns boxes in XYXY format (x1, y1, x2, y2)

**EfficientTAM Input**: (Frame index, boxes XYXY)
- Initializes state on first frame
- Propagates temporal context internally
- Returns mask coordinates as (ys, xs) tuple of numpy arrays

**Mask Representation**: Coordinate lists
- **Not** a dense binary matrix
- Stored as (ys, xs): two arrays of pixel row/column indices
- Compact: only stores foreground pixels
- Converted to binary matrix only when needed (IoU calculation)

**Downsample Option**:
```yaml
run:
  downsample: 1   # Use all frames
  downsample: 2   # Use every 2nd frame (2x speedup)
```

---

## Stage 2: Temporal Deduplication Filter

### Problem Solved

YOLO + EfficientTAM can produce multiple overlapping masks per object per frame:

```
Frame 0:
  Mask A: pixels {(10,20), (10,21), ...}  ← droplet 1
  Mask B: pixels {(10,21), (10,22), ...}  ← same droplet, different bbox prompt

These are ~95% overlap → should be merged
```

### Solution: Forward IoU Filter

```
For each mask M_f at frame f:
  Look ahead w frames: f, f+1, ..., f+w
  
  For each frame in lookahead:
    Compare M_f with all masks in that frame
    If IoU(M_f, other_mask) > match_iou:
      Mark M_f as duplicate
  
  Keep only the highest-confidence mask
```

### Implementation: `stage2_forward_iou_filter()`

**File**: `src/pipeline.py`

**Pseudo-code**:

```python
def stage2_forward_iou_filter(per_frame_masks, window=2, match_iou=0.05):
    """Remove per-frame duplicates via forward IoU."""
    
    filtered_masks = {}
    
    for frame_idx in sorted(per_frame_masks.keys()):
        frame_masks = per_frame_masks[frame_idx]
        
        # Track which masks to keep
        to_keep = []
        
        for local_id, mask_coords in frame_masks.items():
            is_dup = False
            
            # Check against future frames (forward window)
            for future_f in range(frame_idx + 1, min(frame_idx + window + 1, max_frame)):
                if future_f not in per_frame_masks:
                    continue
                
                for future_id, future_coords in per_frame_masks[future_f].items():
                    iou = compute_iou(mask_coords, future_coords)
                    
                    if iou > match_iou:
                        # High overlap → likely duplicate
                        is_dup = True
                        break
            
            if not is_dup:
                to_keep.append((local_id, mask_coords))
        
        filtered_masks[frame_idx] = dict(to_keep)
    
    return filtered_masks
```

### Config Options

```yaml
stage2:
  window: 2              # Frames to look ahead
  mode: "forward_iou"    # forward_iou (faster) or propagate (bidirectional)
  match_iou: 0.05        # 5% IoU → duplicate
```

**Trade-offs**:
- **Smaller window**: Faster, but may miss duplicates spanning many frames
- **Lower match_iou**: More aggressive deduplication, but may merge different objects
- **"propagate" mode**: Slower but more thorough (checks both directions)

---

## Stage 3: Long-Term Tracking

### Problem: Assign Global IDs

```
Frame 0: Masks A, B, C (local IDs 0, 1, 2)
Frame 1: Masks D, E, F (local IDs 0, 1, 2)
...
Frame 2500: Masks X, Y (local IDs 0, 1)

→ Assign global track IDs that connect same droplet across frames
```

### Solution: IoU-Based Greedy Matching

**Key idea**: A droplet at frame *f* that overlaps (IoU > threshold) with a mask at frame *f+1* is the same droplet.

```
Track ID 1: Frame 0→Mask_A, Frame 1→Mask_D, Frame 2→Mask_G, ...
Track ID 2: Frame 0→Mask_B, Frame 1→Mask_E, ...
Track ID 3: Frame 2→Mask_H (new object, no match in earlier frames)
```

### Implementation: `stage3_long_term_tracking()`

**File**: `src/pipeline.py`

**Pseudo-code**:

```python
def stage3_long_term_tracking(per_frame_masks, overlap_thr=0.10, max_skip=0):
    """Assign global track IDs via greedy IoU matching."""
    
    T = {}  # {track_id: {frame_idx: (ys, xs)}}
    next_id = 0
    active_tracks = {}  # {track_id: last_frame_with_mask}
    
    for frame_idx in sorted(per_frame_masks.keys()):
        frame_masks = per_frame_masks[frame_idx]
        matched_masks = set()
        
        # 1. Try to match with active tracks
        for track_id in list(active_tracks.keys()):
            last_frame = active_tracks[track_id]
            gap = frame_idx - last_frame
            
            if gap > max_skip:
                # Track too old → dead
                del active_tracks[track_id]
                continue
            
            # Get previous mask of this track
            prev_mask = T[track_id][last_frame]
            
            # Find best matching mask in current frame
            best_iou = -1
            best_mask_id = None
            
            for mask_id, mask_coords in frame_masks.items():
                if mask_id in matched_masks:
                    continue
                
                iou = compute_iou(prev_mask, mask_coords)
                
                if iou > best_iou:
                    best_iou = iou
                    best_mask_id = mask_id
            
            # If good match found
            if best_iou > overlap_thr:
                T[track_id][frame_idx] = frame_masks[best_mask_id]
                active_tracks[track_id] = frame_idx
                matched_masks.add(best_mask_id)
        
        # 2. Unmatched masks → new tracks
        for mask_id, mask_coords in frame_masks.items():
            if mask_id not in matched_masks:
                T[next_id] = {frame_idx: mask_coords}
                active_tracks[next_id] = frame_idx
                next_id += 1
    
    return T  # {track_id: {frame_idx: (ys, xs)}}
```

### Parameters

```yaml
stage3:
  overlap_thr: 0.10      # IoU threshold to link masks
  min_px: 2              # Drop masks smaller than N pixels
  max_skip: 0            # Allow gaps up to N frames
  kill_after_gap: 6      # Delete tracks absent >N frames
```

**overlap_thr trade-offs**:
- **Higher** (e.g., 0.2): Stricter matching; more separate IDs; less false merges
- **Lower** (e.g., 0.05): Looser matching; fewer IDs; more false merges

**max_skip trade-offs**:
- **0**: No gaps allowed; occlusions → new track
- **3**: Allow occlusions for up to 3 frames
- **6+**: Aggressive relinking; may merge unrelated objects

---

## Multi-GPU Chunked Pipeline

### Architecture

```
Input: 2500 frames, 4 GPUs available

Chunk 1: Frames [0-1000]
  GPU 0 ──┐
          ├─ run_pipeline(0, 1000)
          │  Outputs: chunk_0/T.json, chunk_0/segonly.json
          │
Chunk 2: Frames [998-1998]
  GPU 1 ──┤  (Overlap with chunk 1: frames 998-1000)
          │  Outputs: chunk_1/T.json, chunk_1/segonly.json
          │
Chunk 3: Frames [1996-2500]
  GPU 2 ──┘  (Overlap with chunk 2: frames 1996-1998)
             Outputs: chunk_2/T.json, chunk_2/segonly.json

Merge Phase (single-threaded):
  ↓
1. Load chunk_0/T.json → Global T with IDs (0, 1, 2, ...)
2. Load chunk_1/T.json → Local IDs (0, 1, 2, ...)
   
   For overlap frames (998-1000):
     Compute IoU between chunk_0 and chunk_1 masks
     Build re-ID mapping: chunk_1_id → chunk_0_id
     Apply mapping to all chunk_1 frames
   
   Append chunk_1 frames to global T
   
3. Load chunk_2/T.json
   
   For overlap frames (1996-1998):
     Same re-ID procedure
   
   Append chunk_2 frames to global T

Final Output: Combined T with globally consistent IDs
```

### File Layout

```
outputs/<timestamp>/
├── chunk_0/
│   ├── T.json
│   └── segonly.json
├── chunk_1/
│   ├── T.json
│   └── segonly.json
├── chunk_2/
│   ├── T.json
│   └── segonly.json
└── final/
    ├── tracks.parquet         ← Merged global T
    └── segonly.parquet        ← Merged segonly
```

### Implementation: `src/run_chunks.py`

**Key functions**:

1. **`_make_chunks_by_len(N, m, overlap)`**
   - Split N frames into chunks of size m
   - Each chunk starts `overlap` frames before previous end

2. **`_worker(gpu_id, chunk_start, chunk_end, cfg, ...)`**
   - Child process running on one GPU
   - Calls `run_pipeline()` on chunk
   - Saves outputs to `outputs/<ts>/chunk_X/`

3. **`_merge_chunks(chunks, overlap, ...)`**
   - Re-ID via overlap frame IoU
   - Build global ID mapping
   - Concatenate all chunks into final parquet

---

## Data Format: Tracks & Masks

### Coordinate Representation

**Stored as**: (ys, xs) tuple of numpy arrays

```python
# Example mask: 3x3 region with some pixels
mask_coords = (
    np.array([0, 0, 1, 1, 2]),  # row indices (ys)
    np.array([1, 2, 0, 1, 2])   # col indices (xs)
)

# This represents pixels:
# (0,1), (0,2), (1,0), (1,1), (2,2)
```

**Why this format?**
- Compact: only O(N) space where N = # pixels
- Efficient: numpy array operations are fast
- Natural: matches how we iterate over masks

**Conversion to binary matrix**:

```python
def coords_to_bool(coords, H, W):
    ys, xs = coords
    m = np.zeros((H, W), dtype=bool)
    m[ys, xs] = True
    return m
```

**Conversion to flat indices** (for parquet):

```python
def coords_to_flat(coords, W):
    ys, xs = coords
    return (ys * W + xs).astype(np.int32).tolist()
```

Stored as `list[int]` in parquet to save space.

---

## Parquet Schema

### `tracks_clean.parquet`

After all stages:

```
track_id        int32      Global track ID (1, 2, 3, ...)
frame           int32      Frame index
mask_pixels     list[int]  Flat pixel indices (y*W+x)
area            int32      Number of pixels
y_min, y_max    int32      Bounding box bounds
x_min, x_max    int32
```

**Example row**:
```
track_id=1, frame=0, mask_pixels=[1024, 1025, 2048, 2049, ...], area=142, 
y_min=2, y_max=10, x_min=32, x_max=45
```

### `segonly_clean.parquet`

Weld pool masks (class 0, tracked per-frame only):

```
frame           int32      Frame index
pool_pixels     list[int]  Flat pixel indices
area            int32      Pool area
```

---

## Loss of Information at Each Stage

### Stage 1 Output
- **Per-frame masks** for detected droplets + weld pool
- **Frame-level information** only; no temporal context yet

### Stage 2 Output
- **Deduplicated per-frame masks**
- Loss: Removed duplicates (intentional)

### Stage 3 Output
- **Global tracks with IDs**
- Loss: Masks that didn't match any other frame may be assigned new IDs
- Loss: Very small masks may be dropped (min_px filter)

### Post-Processing Output (Optional)
- **Filtered tracks**
- Loss: Tracks < min_len_frames removed
- Loss: "Yoyo" tracks (brief appearance→disappearance→reappearance) trimmed
- Loss: Falling filter removes upward motion (physics constraint)
- Loss: Bottom stuck filter removes stationary objects at pool edge

---

## Performance Characteristics

### Stage 1: Detection + Segmentation

```
Per frame:
  YOLO inference: ~30-100 ms (depends on frame resolution)
  EfficientTAM: ~50-200 ms
  Total: ~100-300 ms/frame

2500-frame video: 4-12 minutes (single GPU)
```

**Bottleneck**: EfficientTAM memory + computation

### Stage 2: Temporal Filter

```
Quadratic in frame count (worst case):
  For each frame f: compare with ~2*window future frames
  For each pair: compute IoU (O(mask_size))

Typical: ~1% of stage 1 time
```

### Stage 3: Long-Term Tracking

```
Linear in frame count:
  Greedy matching is O(N * M) where N = frames, M = objects/frame

Typical: ~1-5% of stage 1 time
```

### Multi-GPU Speedup

With 4 GPUs + overlap strategy:

```
Single GPU:  2500 frames × 200ms/frame = 500 sec = 8.3 min

4 GPUs:
  - Chunk time: 625 frames × 200ms = 125 sec
  - 4 chunks run in parallel: ~4 chunks take ~2 × 125 sec = 250 sec
  - Merge overhead: ~10 sec
  Total: ~260 sec = 4.3 min (1.9x speedup)

Note: Speedup < 4x due to:
  1. Overlap frames processed multiple times
  2. Merge step is single-threaded
  3. Load imbalance (uneven GPU availability)
```

---

## Checkpointing & Resume

### Current: No built-in checkpointing

The pipeline doesn't save intermediate results during execution. If interrupted, restart from scratch.

**Future improvement**: Save per-frame masks after stage 1 to disk; skip re-computation on resume.

---

## Extension Points

### Add a Custom Post-Processing Filter

**File**: `tools/postprocess_tracks_min.py`

```python
def my_custom_filter(T, cfg):
    """Custom filter."""
    cfg_filter = cfg.get("my_filter", {})
    if not cfg_filter.get("enabled", False):
        return T
    
    T_filtered = {}
    
    for track_id, frames in T.items():
        # Your logic here
        # ...
        
        if keep_this_track:
            T_filtered[track_id] = frames
    
    return T_filtered
```

Register in main:
```python
# src/main.py
T = postprocess_tracks(T, cfg)  # existing
T = my_custom_filter(T, cfg)    # new
```

---

### Add a Custom Model

**File**: Create `src/my_detector.py`

```python
class MyDetector:
    def __init__(self, weights):
        # Load your model
        pass
    
    def detect_xyxy(self, img_path):
        # Return (boxes, classes, confidences)
        pass
```

Update `src/pipeline.py`:
```python
# Replace YOLODetector with MyDetector
det = MyDetector(cfg["my_detector"]["weights"])
```

---

## Debugging Techniques

### Inspect Per-Frame Masks

```python
import json
from pathlib import Path

# After stage 1 + 2
T_json = json.load(open("outputs/latest/chunk_0/T.json"))

# Track 0 over time
track_0 = T_json["0"]  # Keys are frame indices as strings
for frame_str, mask_coords in track_0.items():
    frame_idx = int(frame_str)
    ys, xs = mask_coords
    print(f"Frame {frame_idx}: {len(ys)} pixels")
```

### Visualize a Single Frame

```python
import numpy as np
from PIL import Image

frame_0_img = Image.open("videos/ALS28T3/frame_0000.jpg")
frame_0 = np.array(frame_0_img)

# Overlay masks
H, W = frame_0.shape[:2]

# Mask coords
mask_coords = (np.array([...]), np.array([...]))
ys, xs = mask_coords

overlay = frame_0.copy()
overlay[ys, xs] = [255, 0, 0]  # Red overlay

Image.fromarray(overlay).save("debug_frame_0.png")
```

### Check IoU Computation

```python
from src.utils import iou_masks

mask_a = (np.array([0, 0, 1]), np.array([0, 1, 0]))
mask_b = (np.array([0, 1]), np.array([1, 0]))

iou = iou_masks(mask_a, mask_b, H=3, W=3)
print(f"IoU: {iou:.4f}")
```

---

## Version Notes

**Current Code**: Branch `krishna-simple-reproduction`

**Known Issues**:
- No checkpointing; full re-run required on interrupt
- Overlap merge can be slow for many chunks
- GPU memory usage not optimized for very large frames

**Future Work**:
- Resume from checkpoint
- Streaming IO for very large videos
- Adaptive batch sizing
- Quantization support for edge deployment

---

**Last Updated**: August 2026
