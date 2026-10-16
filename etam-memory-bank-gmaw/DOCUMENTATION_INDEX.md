# HS-WeldCount Documentation Index

Complete reference documentation for the Efficient Track Anything Model (ETAM) on GMAW welding videos.

---

## 📚 Documentation Structure

This project includes four comprehensive guides:

### 1. **SETUP_AND_INSTALLATION.md** ← Start here if you're new
- Python environment setup
- Dependency installation
- GPU/CUDA configuration
- Weight downloads
- Verification checklist
- Troubleshooting common issues

**Read this if you**: Haven't set up the environment yet

---

### 2. **REPRODUCTION_GUIDE.md** ← Detailed walkthrough
- Quick start (TL;DR)
- Full installation steps
- Model weight downloads (Google Drive links)
- Project file structure explained
- Pipeline stages 1-3 explained
- Multi-GPU chunking explained
- Configuration reference
- Running the pipeline
- Output interpretation
- File-by-file working reference
- Troubleshooting

**Read this if you**: Want to understand how each component works, or run the pipeline yourself

---

### 3. **ARCHITECTURE.md** ← Technical deep-dive
- System overview diagram
- Stage-by-stage data flow
- Implementation details (pseudo-code)
- Coordinate representations
- Multi-GPU merge strategy
- Parquet schema
- Performance characteristics
- Extension points for custom work
- Debugging techniques

**Read this if you**: Want to modify code, add custom filters, or understand algorithm details

---

### 4. **DOCUMENTATION_INDEX.md** ← This file
- Navigation guide
- Quick reference
- Decision trees for common tasks

**Read this if you**: Need to find something quickly

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Navigate to project
cd c:\Users\Kashif\ computers\etam-memory-bank-gmaw

# 2. Create & activate environment
python -m venv venv
./venv/Scripts/Activate.ps1  # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Download weights from Google Drive
#    https://drive.google.com/drive/folders/1iY52wo8ZthAMKxm3gh3Hr5jdWVaijZyH?usp=sharing
#    Place: yolov11x/best.pt
#    Place: efficient_track_anything/checkpoints/efficienttam_s.pt

# 5. Edit config
#    configs/pipeline.yaml: update data.video_dir to your frame folder

# 6. Run
python -m src.main --cfg configs/pipeline.yaml

# 7. Check outputs
#    outputs/<timestamp>/final/
```

---

## 🔍 Find Information By Task

### Setting Up

- **I need to install Python packages**
  → SETUP_AND_INSTALLATION.md § Step 2

- **I need to download model weights**
  → SETUP_AND_INSTALLATION.md § Step 3
  → REPRODUCTION_GUIDE.md § Section 2

- **Something is failing during setup**
  → SETUP_AND_INSTALLATION.md § Troubleshooting

- **I want to verify my setup is correct**
  → SETUP_AND_INSTALLATION.md § Step 7

---

### Running the Pipeline

- **Quick run, nothing fancy**
  → REPRODUCTION_GUIDE.md § Quick Start (TL;DR)

- **I want to run on multiple GPUs**
  → REPRODUCTION_GUIDE.md § Section 5 (Multi-GPU Chunking)

- **I want to run only part of a video**
  → REPRODUCTION_GUIDE.md § Section 7.2 (Advanced: Partial Frames)

- **I want paper-exact results (no post-processing)**
  → REPRODUCTION_GUIDE.md § Section 7.4 (Paper-Strict Mode)

- **Something went wrong during a run**
  → REPRODUCTION_GUIDE.md § Section 10 (Troubleshooting)

---

### Understanding the Code

- **What does each Python file do?**
  → REPRODUCTION_GUIDE.md § Section 9 (File-by-File Working Reference)

- **How do the 3 pipeline stages work?**
  → REPRODUCTION_GUIDE.md § Section 4 (Core Pipeline)

- **I want to understand the data structures**
  → ARCHITECTURE.md § Data Format: Tracks & Masks

- **I need to modify the pipeline**
  → ARCHITECTURE.md § Extension Points

- **I want implementation details & pseudo-code**
  → ARCHITECTURE.md § Stages 1-3 Implementation

---

### Working with Outputs

- **What do the output files contain?**
  → REPRODUCTION_GUIDE.md § Section 8 (Output Interpretation)

- **How do I read the parquet files?**
  → REPRODUCTION_GUIDE.md § Section 8.1 (tracks_clean.parquet)
  → ARCHITECTURE.md § Parquet Schema

- **How do I count total droplets?**
  → REPRODUCTION_GUIDE.md § Section 8.1 (Example: read tracks parquet)

- **What does the MP4 video show?**
  → REPRODUCTION_GUIDE.md § Section 8.6 (video_out.mp4)

---

### Configuration & Tuning

- **What are all the config options?**
  → REPRODUCTION_GUIDE.md § Section 6 (Configuration Guide)

- **How do I adjust detection sensitivity?**
  → REPRODUCTION_GUIDE.md § Section 6 (yolo.conf, yolo.iou)

- **How do I reduce memory usage?**
  → SETUP_AND_INSTALLATION.md § Troubleshooting: CUDA out of memory

- **What should I set for my GPU setup?**
  → REPRODUCTION_GUIDE.md § Section 6 (parallel section)

---

### Advanced Topics

- **How does multi-GPU chunking work?**
  → REPRODUCTION_GUIDE.md § Section 5
  → ARCHITECTURE.md § Multi-GPU Chunked Pipeline

- **What's the performance (speed, memory)?**
  → ARCHITECTURE.md § Performance Characteristics

- **How do I add a custom post-processing filter?**
  → ARCHITECTURE.md § Extension Points (Add Custom Filter)

- **How do I debug a single frame?**
  → ARCHITECTURE.md § Debugging Techniques

---

## 📋 Decision Trees

### "I want to run the pipeline"

```
Do you have the weights files?
├─ No  → SETUP_AND_INSTALLATION.md § Step 3
└─ Yes → Continue

Do you want single GPU or multi-GPU?
├─ Single → REPRODUCTION_GUIDE.md § Section 7.1
└─ Multi  → REPRODUCTION_GUIDE.md § Section 7.3

Do you want paper-strict results or with extras?
├─ Paper-strict → REPRODUCTION_GUIDE.md § 7.4
└─ With extras  → REPRODUCTION_GUIDE.md § 7.1

Are you running all frames or a subset?
├─ All   → Just run: python -m src.main --cfg configs/pipeline.yaml
└─ Subset → Use --frame-start and --frame-end flags
```

---

### "I want to modify the code"

```
What do you want to modify?

1. Detection (YOLO)
   → Replace YOLODetector in src/models.py
   → ARCHITECTURE.md § Add a Custom Model

2. Segmentation (EfficientTAM)
   → Replace EfficientTAMTracker in src/models.py
   → Note: requires compatible checkpoint
   → ARCHITECTURE.md § Add a Custom Model

3. Post-processing (filters)
   → Edit tools/postprocess_tracks_min.py
   → ARCHITECTURE.md § Add Custom Post-Processing Filter

4. Tracking (Stage 3 logic)
   → Edit stage3_long_term_tracking() in src/pipeline.py
   → ARCHITECTURE.md § Stage 3: Long-Term Tracking

5. Temporal deduplication (Stage 2)
   → Edit stage2_forward_iou_filter() in src/pipeline.py
   → ARCHITECTURE.md § Stage 2: Temporal Deduplication
```

---

### "Something is broken"

```
Where is the error?

1. Installation
   → SETUP_AND_INSTALLATION.md § Troubleshooting

2. Running the pipeline
   → REPRODUCTION_GUIDE.md § Section 10 (Troubleshooting)

3. GPU/CUDA issues
   → SETUP_AND_INSTALLATION.md § Issue: torch.cuda.is_available() returns False

4. Low accuracy / wrong results
   → Adjust config: REPRODUCTION_GUIDE.md § Section 6
   → Check if post-processing is helping/hurting: try paper-strict mode
   → Debug single frame: ARCHITECTURE.md § Debugging Techniques
```

---

## 📖 Reading Recommendations by Role

### **Data Scientist / Researcher**
Read in order:
1. REPRODUCTION_GUIDE.md § Quick Start + Section 4 (Pipeline overview)
2. REPRODUCTION_GUIDE.md § Section 8 (Outputs)
3. REPRODUCTION_GUIDE.md § Section 10 (Troubleshooting)

Time investment: ~30 minutes

---

### **Software Engineer / ML Engineer**
Read in order:
1. SETUP_AND_INSTALLATION.md (full)
2. REPRODUCTION_GUIDE.md § Sections 1-9
3. ARCHITECTURE.md (full)

Time investment: ~2 hours

---

### **System Administrator / DevOps**
Read in order:
1. SETUP_AND_INSTALLATION.md § Steps 1-2, 7
2. REPRODUCTION_GUIDE.md § Section 5 (Multi-GPU)
3. ARCHITECTURE.md § Performance Characteristics

Time investment: ~1 hour

---

### **Quick Reference (Just Run It)**
1. SETUP_AND_INSTALLATION.md § Steps 1-6
2. REPRODUCTION_GUIDE.md § Quick Start (TL;DR)
3. Run!

Time investment: ~20 minutes

---

## 🔧 Configuration Quick Reference

| Parameter | Default | Use Case |
|-----------|---------|----------|
| `yolo.conf` | 0.15 | Lower = more detections; adjust if missing droplets |
| `yolo.iou` | 0.30 | Lower = separate boxes; adjust for NMS tuning |
| `stage2.match_iou` | 0.05 | Lower = aggressive deduplication |
| `stage3.overlap_thr` | 0.10 | Lower = loose tracking; adjust if incorrect merges |
| `frames_per_part` | 1000 | Smaller = less memory; adjust if GPU OOM |
| `downsample` | 1 | 1 = all frames; 2 = every 2nd (faster but lower quality) |
| `device` | "auto" | "auto", "cuda", or "cpu" |

---

## 📊 Output Files Quick Reference

| File | Format | Contains |
|------|--------|----------|
| `tracks_clean.parquet` | Parquet | All droplet tracks (ID, frame, mask pixels) |
| `centroids_clean.parquet` | Parquet | Track centroids (x, y per frame) |
| `droplet_count_summary.json` | JSON | Aggregate statistics (total count, etc.) |
| `droplet_count_per_frame.csv` | CSV | Per-frame droplet counts |
| `overlays_post/*.png` | PNG | Frames with drawn masks + IDs |
| `video_out.mp4` | MP4 | Annotated video (30 fps by default) |

---

## 🎯 Common Workflows

### Workflow 1: First-time run

```
1. SETUP_AND_INSTALLATION.md § Steps 1-4
   └─ Install + verify
2. Edit configs/pipeline.yaml
   └─ Point to your frames
3. python -m src.main --cfg configs/pipeline.yaml
   └─ Run
4. REPRODUCTION_GUIDE.md § Section 8
   └─ Interpret outputs
```

---

### Workflow 2: Hyperparameter tuning

```
1. Run once with default config
2. REPRODUCTION_GUIDE.md § Section 6
   └─ Adjust parameters
3. REPRODUCTION_GUIDE.md § Section 7 (Advanced: Partial Frames)
   └─ Test on subset
4. Iterate until happy
5. Full run
```

---

### Workflow 3: Paper reproduction

```
1. REPRODUCTION_GUIDE.md § Section 7.4 (Paper-Strict Mode)
   └─ Disable post-processing
2. Run full pipeline
3. Count unique IDs in tracks_clean.parquet
   └─ This is your droplet count
```

---

### Workflow 4: Multi-GPU processing

```
1. SETUP_AND_INSTALLATION.md § Step 2.2
   └─ GPU-accelerated torch
2. REPRODUCTION_GUIDE.md § Section 6
   └─ Edit parallel config
3. python -m src.main --cfg configs/pipeline.yaml
   └─ Will auto-parallelize
```

---

## 📞 Getting Help

**For installation issues**:
→ SETUP_AND_INSTALLATION.md § Troubleshooting

**For pipeline execution issues**:
→ REPRODUCTION_GUIDE.md § Section 10 (Troubleshooting)

**For understanding algorithms**:
→ ARCHITECTURE.md § Stages 1-3 Implementation

**For modifying code**:
→ ARCHITECTURE.md § Extension Points
→ REPRODUCTION_GUIDE.md § Section 9 (File-by-File)

---

## 📝 File Manifest

```
Documentation:
├── DOCUMENTATION_INDEX.md        ← You are here
├── SETUP_AND_INSTALLATION.md     ← Start with environment setup
├── REPRODUCTION_GUIDE.md         ← Complete walkthrough + reference
├── ARCHITECTURE.md               ← Technical deep-dive
└── README.md                     ← Project overview

Code:
├── src/
│   ├── main.py                   ← Orchestrator entry point
│   ├── pipeline.py               ← 3-stage core logic
│   ├── models.py                 ← Model wrappers
│   ├── run_chunks.py             ← Multi-GPU scheduler
│   └── utils.py                  ← Helper functions
├── tools/
│   ├── postprocess_tracks_min.py ← Optional post-processing
│   ├── count_droplets.py         ← Droplet statistics
│   └── render_overlays_min.py    ← Visualization
├── efficient_track_anything/     ← Segmentation package
├── yolov11x/                     ← YOLO weights (download)
├── configs/
│   ├── pipeline.yaml             ← Main config file
│   └── efficienttam/             ← Model configs
├── file2mp4.py                   ← Frame → MP4 encoder
└── requirements.txt              ← Python dependencies
```

---

## Version & Status

**Current Branch**: `krishna-simple-reproduction`

**Last Updated**: August 2026

**Status**: ✅ Active (fully functional)

**Known Limitations**:
- No checkpoint/resume (full re-run on interrupt)
- Overlap merge can be slow for many GPU chunks
- GPU memory usage not optimized for very large frames

---

**Questions?** Check the relevant guide above or see REPRODUCTION_GUIDE.md § Troubleshooting.

**Ready to start?** → SETUP_AND_INSTALLATION.md § Step 1
