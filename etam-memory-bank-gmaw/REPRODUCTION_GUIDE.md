# HS-WeldCount: Complete Reproduction & Working Guide

**Project**: Droplet Tracking, Segmentation, and Counting in High-Speed GMAW Welding Videos

**Reference**: 
- Xiong, Y. et al. **Efficient Track Anything.** [arXiv:2411.18933](https://arxiv.org/abs/2411.18933)
- Bondi, E. et al. **CountVid: Open-World Object Counting in Videos.** [arXiv:2506.15368](https://arxiv.org/abs/2506.15368)

---

## Quick Start (TL;DR)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download model weights to the correct folders
#    (See "2. Download Model Weights" section below)

# 3. Update config with your video frame directory
#    Edit configs/pipeline.yaml: data.video_dir = "/path/to/frames"

# 4. Run the full pipeline
python -m src.main --cfg configs/pipeline.yaml

# 5. Check outputs in outputs/<timestamp>/final/
```

---

## 1. Installation & Environment Setup

### 1.1 Prerequisites

- **Python**: 3.9+ (tested on 3.10-3.11)
- **GPU**: NVIDIA GPU with CUDA support (strongly recommended; CPU mode is very slow)
- **RAM**: 16GB+ recommended for multi-GPU chunking

### 1.2 Install Dependencies

```bash
# Clone/navigate to repo
cd c:\Users\Kashif\ computers\etam-memory-bank-gmaw

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install base requirements
pip install -r requirements.txt

# If you have a CUDA-capable GPU, install the matching torch wheel:
# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Check CUDA version: nvidia-smi
```

### 1.3 Verify Installation

```python
import torch
print(torch.cuda.is_available())  # Should print True if GPU detected
print(torch.cuda.device_count())  # Number of GPUs
```

---

## 2. Download Model Weights

### 2.1 Required Weights

This project uses **two** pre-trained models:

| Model | Purpose | File | Destination |
|-------|---------|------|-------------|
| **YOLO v11x** | Frame-by-frame droplet detection | `best.pt` | `yolov11x/best.pt` |
| **EfficientTAM-S** | Segmentation & temporal propagation | `efficienttam_s.pt` | `efficient_track_anything/checkpoints/efficienttam_s.pt` |

### 2.2 Download from Google Drive

📥 **Google Drive Link**: https://drive.google.com/drive/folders/1iY52wo8ZthAMKxm3gh3Hr5jdWVaijZyH?usp=sharing

**Steps**:

1. Visit the Google Drive link
2. Download:
   - `best.pt` (YOLO detector)
   - `efficienttam_s.pt` (EfficientTAM segmenter)

3. Create the destination folders if they don't exist:
   ```bash
   mkdir -p yolov11x
   mkdir -p efficient_track_anything/checkpoints
   ```

4. Place files:
   - `best.pt` → `yolov11x/best.pt`
   - `efficienttam_s.pt` → `efficient_track_anything/checkpoints/efficienttam_s.pt`

### 2.3 Alternative: Download EfficientTAM via Script (HuggingFace)

EfficientTAM checkpoints are also available on HuggingFace. Run the provided script:

```bash
cd efficient_track_anything/checkpoints
bash download_checkpoints.sh
cd ../../
```

This will download multiple EfficientTAM variants (TI, S, etc.). For this project, we only need **efficienttam_s.pt**.

### 2.4 Verify Downloads

```bash
# Check file sizes (both should be several MB to GB)
ls -lh yolov11x/best.pt
ls -lh efficient_track_anything/checkpoints/efficienttam_s.pt
```

---

## 3. Project Structure & File Explanations

```
c:\Users\Kashif computers\etam-memory-bank-gmaw/
├── src/                          # Core pipeline implementation
│   ├── __init__.py
│   ├── main.py                   # Orchestrator: runs all stages + post-processing
│   ├── pipeline.py               # Stages 1-3 logic (detect/segment/track)
│   ├── models.py                 # Model wrappers (YOLODetector, EfficientTAMTracker)
│   ├── run_chunks.py             # Multi-GPU chunked scheduler
│   └── utils.py                  # Helpers (IO, mask/IoU ops, transforms)
│
├── tools/                        # Post-processing and visualization
│   ├── postprocess_tracks_min.py # Temporal filtering + droplet-specific filters
│   ├── count_droplets.py         # Aggregate droplet counts & stats
│   └── render_overlays_min.py    # Draw masks + IDs onto video frames
│
├── configs/                      # Configuration files
│   ├── pipeline.yaml             # Main config (data paths, model params, filters)
│   └── efficienttam/             # EfficientTAM model configs
│       ├── efficienttam_s.yaml
│       ├── efficienttam_ti.yaml
│       └── ... (other variants)
│
├── efficient_track_anything/     # EfficientTAM package (segmentation + tracking)
│   ├── __init__.py
│   ├── build_efficienttam.py     # Builder: loads model + config
│   ├── efficienttam_video_predictor.py
│   ├── checkpoints/              # ← Place efficienttam_s.pt here
│   │   └── download_checkpoints.sh
│   ├── modeling/                 # EfficientTAM architecture
│   │   ├── efficienttam_base.py
│   │   ├── memory_attention.py
│   │   ├── mask_decoder.py
│   │   └── ... (components)
│   └── utils/                    # Utilities
│
├── yolov11x/                     # ← Place best.pt (YOLO detector) here
│
├── file2mp4.py                   # Frames → MP4 video
├── requirements.txt              # Python dependencies
├── README.md                     # High-level overview
└── outputs/                      # Created at runtime
    └── <timestamp>/
        ├── final/                # ← Main outputs land here
        │   ├── tracks_clean.parquet
        │   ├── centroids_clean.parquet
        │   ├── droplet_count_summary.json
        │   ├── droplet_count_per_frame.csv
        │   └── overlays_post/
        │       └── *.png         # Annotated frames
        └── video_out.mp4         # Final video
```

---

## 4. Core Pipeline: The 3 Stages

### **Stage 1: Per-Frame Detection + Segmentation**

**File**: `src/pipeline.py` → `stage1_detect_and_segment()`

**What it does**:
1. Load each video frame (independently)
2. Run YOLO to detect bounding boxes for droplets (class 1) and weld pool (class 0)
3. Pass bboxes to EfficientTAM for mask prediction
4. Collect per-frame masks

**Key Parameters** (`configs/pipeline.yaml`):
```yaml
yolo:
  weights: "./yolov11x/best.pt"
  conf: 0.15              # Confidence threshold
  iou: 0.30               # NMS IoU threshold
  classes:
    track: [1]            # Droplets (full tracking)
    segment_only: [0]     # Weld pool (per-frame only)

efficienttam:
  cfg: "configs/efficienttam/efficienttam_s.yaml"
  ckpt: "efficient_track_anything/checkpoints/efficienttam_s.pt"
```

**Output**: Per-frame masks stored in memory; written as JSON

---

### **Stage 2: Temporal Filter**

**File**: `src/pipeline.py` → `stage2_forward_iou_filter()`

**What it does**:
1. For each mask at frame *f*, compare it with masks in a window of *w* nearby frames
2. Masks with high IoU (> threshold) are considered "same object"
3. Keep highest-confidence mask, drop duplicates
4. Reduces noise and per-frame artifacts

**Key Parameters** (`configs/pipeline.yaml`):
```yaml
stage2:
  window: 2               # Frames to look back/forward
  mode: "forward_iou"    # forward_iou or propagate (bidirectional)
  match_iou: 0.05        # IoU threshold for match
```

**Why?** YOLO + EfficientTAM can produce multiple overlapping masks per object per frame. This filter cleans it up.

**Output**: Deduplicated per-frame masks

---

### **Stage 3: Long-Term Tracking + Discovery**

**File**: `src/pipeline.py` → `stage3_long_term_tracking()`

**What it does**:
1. Build a global track ID for each droplet across the entire video
2. Link masks across frames using IoU overlap
3. Assign same ID if overlap > threshold
4. Discover new objects (first appearance, reappearance after occlusion)
5. Handle gaps (object disappears for *max_skip* frames, then reappears)

**Key Parameters** (`configs/pipeline.yaml`):
```yaml
stage3:
  overlap_thr: 0.10      # IoU threshold to link masks across frames
  min_px: 2              # Extra: minimum mask size (pixels) to keep
  max_skip: 0            # Max frames to skip before breaking track
  kill_after_gap: 6      # Extra: delete track if absent > N frames
```

**Output**: Global ID assignments; each mask has a unique ID across the video

---

### **Pipeline Invocation**

**File**: `src/main.py`

This orchestrator runs all stages in sequence:

```python
# 1. Run core pipeline (chunked if GPU config set)
if use_chunks:
    # Multi-GPU: split video into overlapping chunks
    run_chunks(...) 
else:
    # Single GPU: run entire video at once
    run_pipeline(cfg, ...)

# 2. Post-process (temporal filters, droplet-specific rules)
postprocess_tracks_min.py

# 3. Count droplets
count_droplets.py

# 4. Render overlays (optional: draw masks on frames)
render_overlays_min.py

# 5. Make MP4 (optional: combine frames into video)
file2mp4.py
```

---

## 5. Multi-GPU Chunked Pipeline

**File**: `src/run_chunks.py`

### How It Works

For large videos, the pipeline can split frames into overlapping chunks and process each on a separate GPU in parallel:

```
Video frames: [0 ... 999 ... 1999 ... 2999]

Chunk 1: frames [0-1000]   → GPU 0
Chunk 2: frames [998-1998] → GPU 1  (overlaps by 3 frames with Chunk 1)
Chunk 3: frames [1996-2999]→ GPU 0  (overlaps by 3 frames with Chunk 2)

After processing each chunk independently:
Merge by re-ID'ing overlap frames via mutual-best IoU
```

### Configuration

```yaml
parallel:
  gpus: "auto"              # "auto" = all GPUs; or list like [0, 1, 2]
  max_workers_per_gpu: 1    # Processes per GPU
  overlap: 3                # Shared frames for ID stitching
  strategy: "fixed_len"     # or "per_gpu"
  frames_per_part: 1000     # Chunk size
```

### When to Use

- **Single GPU**: Set `strategy: ""` or `strategy: null` or use `--no-chunks`
- **Multiple GPUs**: Set `strategy: "fixed_len"` or `"per_gpu"`

---

## 6. Configuration Guide (`configs/pipeline.yaml`)

### Data Paths

```yaml
data:
  video_dir: "/home/lucas_ccwj/projects/Software_v0.1/ALS28T3"  # ← Update to your frame folder
  output_root: "./outputs"   # Where results land
```

**Note**: All paths are resolved relative to the repo root, not the `configs/` folder.

### Stage-Specific Parameters

**Detection (Stage 1)**:
```yaml
yolo:
  weights: "./yolov11x/best.pt"
  conf: 0.15         # Lower = more detections (more false positives)
  iou: 0.30          # Lower = more separate boxes (less NMS)
  classes:
    track: [1]       # Classes to fully track
    segment_only: [0] # Classes to mask but not track
```

**Segmentation** (used per-frame in Stage 1):
```yaml
efficienttam:
  cfg: "configs/efficienttam/efficienttam_s.yaml"
  ckpt: "efficient_track_anything/checkpoints/efficienttam_s.pt"
```

**Temporal Filter** (Stage 2):
```yaml
stage2:
  window: 2          # Look back/forward 2 frames
  mode: "forward_iou" # Only look forward (faster)
  match_iou: 0.05    # Masks with >5% IoU are same object
```

**Tracking** (Stage 3):
```yaml
stage3:
  overlap_thr: 0.10  # 10% IoU to link across frames
  min_px: 2          # Drop masks smaller than 2 pixels
  max_skip: 0        # Don't allow gaps
  kill_after_gap: 6  # Remove tracks absent >6 frames
```

**Post-Processing** (Optional, not in paper):
```yaml
postprocess:
  enabled: true
  min_len_frames: 6          # Drop tracks shorter than 6 frames
  yoyo_trim: true            # Remove start/end noise
  falling_filter: true       # Physics filter: droplets fall
  bottom_stuck_filter: true  # Remove particles stuck at bottom
  reindex_ids: true          # Renumber IDs for readability
```

**Output** (Optional):
```yaml
output:
  render_overlays: true   # Draw masks on frames
  make_mp4: true          # Encode frames to MP4
  fps: 30                 # MP4 frame rate
```

**Runtime**:
```yaml
run:
  downsample: 1           # 1 = use all frames; 2 = use every 2nd frame
  save_overlays: true     # Save visualization during pipeline
  device: "auto"          # "cuda", "cpu", or "auto"
```

---

## 7. Running the Full Pipeline

### 7.1 Quick Run (Single GPU, All Frames)

```bash
python -m src.main --cfg configs/pipeline.yaml
```

**Time**: ~5-30 min depending on video length and GPU

**Outputs**:
```
outputs/<timestamp>/
├── final/
│   ├── tracks_clean.parquet          # Track IDs + mask coordinates
│   ├── centroids_clean.parquet       # Centroid XY per mask
│   ├── droplet_count_summary.json    # Total count
│   ├── droplet_count_per_frame.csv   # Per-frame stats
│   └── overlays_post/
│       ├── frame_0000.png
│       ├── frame_0001.png
│       └── ...
└── video_out.mp4                      # Annotated video
```

### 7.2 Advanced: Partial Frames

```bash
# Only process frames 100-200
python -m src.main --cfg configs/pipeline.yaml --frame-start 100 --frame-end 200
```

### 7.3 Advanced: Multi-GPU

Edit `configs/pipeline.yaml`:

```yaml
parallel:
  gpus: "auto"           # Or [0, 1, 2, 3]
  strategy: "fixed_len"
  frames_per_part: 1000
```

Then:
```bash
python -m src.main --cfg configs/pipeline.yaml
```

The orchestrator will automatically:
1. Split video into 1000-frame chunks
2. Run each on a different GPU in parallel
3. Stitch results via overlap frame re-IDing

### 7.4 Paper-Strict Mode

To reproduce **exactly** the paper's methodology (no post-processing, no visualization):

```yaml
postprocess:
  enabled: false
output:
  render_overlays: false
  make_mp4: false
```

Run:
```bash
python -m src.main --cfg configs/pipeline.yaml
```

**Result**: `tracks_clean.parquet` contains raw droplet tracks from the 3-stage pipeline. Count = number of unique IDs.

---

## 8. Output Interpretation

### 8.1 `tracks_clean.parquet`

Parquet table with columns:

| Column | Type | Description |
|--------|------|-------------|
| `track_id` | int | Global droplet ID (1, 2, 3, ...) |
| `frame` | int | Frame number |
| `mask_pixels` | list[int] | Flat pixel indices (y*W+x) of mask |
| `area` | int | Number of pixels in mask |
| `y_min, y_max, x_min, x_max` | int | Bounding box |

**Example use**:
```python
import pandas as pd
import numpy as np

tracks = pd.read_parquet("outputs/<timestamp>/final/tracks_clean.parquet")

# Count unique droplets
num_droplets = tracks['track_id'].nunique()
print(f"Total droplets: {num_droplets}")

# Track duration (frames)
track_lengths = tracks.groupby('track_id').size()
print(f"Avg track length: {track_lengths.mean():.1f} frames")
```

### 8.2 `centroids_clean.parquet`

Centroid (center of mass) of each mask:

| Column | Type |
|--------|------|
| `track_id` | int |
| `frame` | int |
| `y_center` | float |
| `x_center` | float |

### 8.3 `droplet_count_summary.json`

```json
{
  "total_droplets": 427,
  "total_frames": 2500,
  "avg_droplets_per_frame": 1.2,
  "max_droplets_in_frame": 5,
  "min_droplets_in_frame": 0
}
```

### 8.4 `droplet_count_per_frame.csv`

```csv
frame,droplet_count,active_track_ids
0,2,"[1, 2]"
1,2,"[1, 2]"
2,3,"[1, 2, 3]"
...
```

### 8.5 `overlays_post/*.png`

Annotated frames with:
- Colored masks (one color per droplet ID)
- Track IDs labeled
- Weld pool highlighted

Useful for **visual validation** of tracks.

### 8.6 `video_out.mp4`

Full video with all overlays. Open in any video player.

---

## 9. File-by-File Working Reference

### `src/main.py`

**Purpose**: High-level orchestrator.

**Flow**:
```
1. Parse CLI args
2. Load YAML config
3. Resolve paths (relative to repo root)
4. If chunked strategy: run_chunks()
   Else: run_pipeline()
5. Post-process (temporal filters)
6. Count droplets
7. (Optional) Render overlays
8. (Optional) Encode MP4
```

**Entry point**: `python -m src.main --cfg <config>`

---

### `src/pipeline.py`

**Purpose**: Core 3-stage logic.

**Functions**:

| Function | Stage | Input | Output |
|----------|-------|-------|--------|
| `stage1_detect_and_segment()` | 1 | Frame paths, YOLO cfg, EfficientTAM cfg | Per-frame masks |
| `stage2_forward_iou_filter()` | 2 | Per-frame masks, IoU threshold | Deduplicated masks |
| `stage3_long_term_tracking()` | 3 | Deduplicated masks | Global IDs (tracks) |
| `run_pipeline()` | 1-3 | Config dict | Parquet files + metadata |

**Key**: Each stage is independent and can be tested separately.

---

### `src/models.py`

**Purpose**: Model wrappers.

| Class | Wraps | Method |
|-------|-------|--------|
| `YOLODetector` | Ultralytics YOLO | `detect_xyxy()`, `detect_by_class()` |
| `EfficientTAMTracker` | EfficientTAM | `segment_boxes_at_frame()`, `propagate_one_frame()` |

**Example**:
```python
from src.models import YOLODetector, EfficientTAMTracker

yolo = YOLODetector("yolov11x/best.pt", conf=0.15)
boxes, classes, confs = yolo.detect_xyxy("frame_001.jpg")

etam = EfficientTAMTracker(
    "configs/efficienttam/efficienttam_s.yaml",
    "efficient_track_anything/checkpoints/efficienttam_s.pt"
)
masks = etam.segment_boxes_at_frame(frame_idx=0, boxes_xyxy=boxes)
```

---

### `src/run_chunks.py`

**Purpose**: Multi-GPU scheduler for chunked processing.

**Functions**:

| Function | Purpose |
|----------|---------|
| `_make_chunks_by_len()` | Split video into overlapping chunks |
| `_worker()` | Child process: runs `run_pipeline()` on one chunk |
| `_merge_chunks()` | Stitch chunk outputs via overlap frame re-IDing |
| `main()` | Orchestrate multi-GPU run |

**Config**:
```yaml
parallel:
  gpus: "auto" | [0, 1, 2] | "0,1,2"
  strategy: "fixed_len" | "per_gpu" | null
  frames_per_part: 1000
  overlap: 3
```

---

### `src/utils.py`

**Purpose**: Utilities.

| Function | What |
|----------|------|
| `load_frame_names()` | Read `.jpg/.png` files from a folder |
| `ensure_run_dir()` | Create output folder structure |
| `iou_masks()` | Compute IoU between two binary masks |
| `bbox_from_mask_bool()` | Get bounding box from mask |
| `save_T_json()` | Persist tracks to JSON (used by `run_chunks` merge) |

---

### `tools/postprocess_tracks_min.py`

**Purpose**: Optional temporal + domain-specific filtering (NOT in paper).

**Filters**:

1. **Min length**: Drop tracks shorter than *N* frames
2. **Yoyo trim**: Remove start/end noise (short disconnected pieces)
3. **Falling filter**: Physics constraint (droplets fall, not go up)
4. **Bottom stuck filter**: Remove particles stuck at weld pool bottom

**Config**:
```yaml
postprocess:
  enabled: true
  min_len_frames: 6
  yoyo_trim: true
  falling_filter: true
  bottom_stuck_filter: true
  reindex_ids: true
```

**Note**: Disable for paper-strict mode.

---

### `tools/count_droplets.py`

**Purpose**: Aggregate statistics from tracks.

**Output**: 
- `droplet_count_summary.json` (total, per-frame averages, min/max)
- `droplet_count_per_frame.csv` (per-frame counts)

---

### `tools/render_overlays_min.py`

**Purpose**: Draw masks + IDs onto frames.

**Input**: Original frame folder + tracks parquet

**Output**: PNG frames in `overlays_post/`

**Config**:
```yaml
output:
  render_overlays: true
```

---

### `file2mp4.py`

**Purpose**: Combine PNG frames into MP4.

**Input**: Overlaid PNG frames

**Output**: `video_out.mp4`

**Config**:
```yaml
output:
  make_mp4: true
  fps: 30
```

---

### `efficient_track_anything/`

**Purpose**: EfficientTAM package (segmentation + temporal propagation).

**Key Files**:

| File | Purpose |
|------|---------|
| `build_efficienttam.py` | Load model from YAML config + checkpoint |
| `efficienttam_video_predictor.py` | Main predictor class |
| `modeling/efficienttam_base.py` | EfficientTAM architecture |
| `modeling/memory_encoder.py` | Temporal memory (tracks state across frames) |
| `modeling/memory_attention.py` | Attention over memory |
| `configs/efficienttam_s.yaml` | Config for small model |

**Usage** (internal):
```python
from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

predictor = build_efficienttam_video_predictor(
    cfg_path="configs/efficienttam/efficienttam_s.yaml",
    ckpt_path="efficient_track_anything/checkpoints/efficienttam_s.pt",
    device="cuda"
)

# Initialize with frame 0 and bboxes
predictor.init_state(image_path, bboxes)

# Propagate to frame 1, 2, ...
predictor.propagate_one_frame(image_path)
```

---

## 10. Troubleshooting

### Issue: "Module not found: efficient_track_anything"

**Solution**:
```bash
# Make sure you're in the repo root
cd c:\Users\Kashif\ computers\etam-memory-bank-gmaw

# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce batch size (increase `frames_per_part` or reduce `max_workers_per_gpu`)
2. Use smaller model variant (`efficienttam_ti` instead of `efficienttam_s`)
3. Enable CPU-only mode (slow): set `device: "cpu"` in config
4. Use fewer GPUs

---

### Issue: "No frames found in video_dir"

**Solution**:
- Ensure `data.video_dir` in `configs/pipeline.yaml` points to a folder with `.jpg` or `.png` files
- Frames must be named consistently (e.g., `frame_0000.jpg`, `img_001.png`)

---

### Issue: "Weights file not found"

**Solutions**:
- Verify files are at:
  - `yolov11x/best.pt`
  - `efficient_track_anything/checkpoints/efficienttam_s.pt`
- Check file sizes (should be several MB to GB, not KB)
- Re-download from Google Drive if corrupted

---

### Issue: "Slow on CPU / want GPU acceleration"

**Solution**:
1. Check CUDA installation:
   ```bash
   nvidia-smi  # Should show GPUs
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Install correct torch wheel for your CUDA version:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
3. Set `device: "cuda"` in config

---

## 11. Reproducibility Notes

### For Paper-Strict Results

```yaml
# configs/pipeline.yaml

# Disable all optional extras
postprocess:
  enabled: false

output:
  render_overlays: false
  make_mp4: false

# Keep stages 1-3 as-is
```

Run:
```bash
python -m src.main --cfg configs/pipeline.yaml
```

Final count = unique `track_id` values in `tracks_clean.parquet`.

---

### For Visual Inspection

Keep all visualizations:
```yaml
output:
  render_overlays: true
  make_mp4: true
  fps: 30
```

Open `outputs/<timestamp>/video_out.mp4` in any video player.

---

## 12. Quick Reference: Key Commands

```bash
# Install
pip install -r requirements.txt

# Download weights (manual or via script)
# See Section 2

# Edit config
nano configs/pipeline.yaml

# Run full pipeline
python -m src.main --cfg configs/pipeline.yaml

# Partial frames
python -m src.main --cfg configs/pipeline.yaml --frame-start 100 --frame-end 200

# Disable chunks (single GPU)
python -m src.main --cfg configs/pipeline.yaml --no-chunks

# Custom output location
python -m src.main --cfg configs/pipeline.yaml --force-run-dir /custom/path

# Use custom MP4 output path
python -m src.main --cfg configs/pipeline.yaml --video-out /path/to/out.mp4
```

---

## 13. Example: Full Workflow

```bash
# 1. Setup
cd c:\Users\Kashif\ computers\etam-memory-bank-gmaw
python -m venv venv
source venv/Scripts/activate

# 2. Install
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Download weights
#    (Manually via Google Drive or via script)
#    Expected locations:
#    - yolov11x/best.pt
#    - efficient_track_anything/checkpoints/efficienttam_s.pt

# 4. Prepare video frames
#    Create a folder with JPG/PNG frames
mkdir videos/ALS28T3
# ... copy frames into videos/ALS28T3/

# 5. Update config
# Edit configs/pipeline.yaml:
# data.video_dir: "./videos/ALS28T3"

# 6. Run
python -m src.main --cfg configs/pipeline.yaml

# 7. Check outputs
ls outputs/<latest_timestamp>/final/
# Should see:
# - tracks_clean.parquet
# - centroids_clean.parquet
# - droplet_count_summary.json
# - droplet_count_per_frame.csv
# - overlays_post/ (PNG frames)
# - video_out.mp4
```

---

## 14. References & Resources

**Papers**:
- **EfficientTAM**: Xiong et al. (2024) — Segmentation and temporal tracking
- **CountVid**: Bondi et al. (2025) — Open-world object counting in videos
- **SAM2**: Meta's foundational work on segmentation

**Code**:
- [EfficientTAM on GitHub](https://github.com/abriefcaseof/efficient-track-anything)
- [YOLOv11 on GitHub](https://github.com/ultralytics/ultralytics)

**Hardware**:
- NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- torch installation: https://pytorch.org/get-started/locally/

---

**Last Updated**: August 2026  
**Project Status**: Active  
**Maintainer**: lucas_ccwj (ccwj@ualberta.ca)
