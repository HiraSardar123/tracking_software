# Tracking Software — GMAW Droplet Tracking and Counting

Software for detecting, segmenting, tracking and counting metal droplets in
high-speed gas metal arc welding (GMAW) video.

The pipeline has three stages:

1. **YOLOv11x** detects droplets in each frame.
2. **EfficientTAM** segments them and propagates the masks.
3. A temporal filter and long-term tracker keep each droplet's identity and find new droplets as they appear.

## Repository layout

```
tracking_software/
├── README.md                    # this file
├── .gitignore                   # keeps all data, media and weights out of git
└── etam-memory-bank-gmaw/       # the tracking/counting code
    ├── src/                     # pipeline core (stages 1-3, chunked multi-GPU runner)
    ├── tools/                   # post-processing, counting, overlays, benchmark,
    │                            #   annotation tracker server
    ├── configs/                 # pipeline.yaml
    ├── efficient_track_anything/  # EfficientTAM segmenter package
    ├── yolov11x/                # detector weights go here (not in git)
    ├── file2mp4.py              # frames -> mp4
    └── *.md                     # architecture, setup, reproduction guide, etc.
```

**This repository contains code only.** Everything below is kept out by
`.gitignore` and must be obtained separately:

| Not in git | What it is |
|---|---|
| `tracking_data/**/*.mp4`, `*.csv`, `*.png` | Video clips, `assignment.csv` sheets and example images (only the READMEs and the `vvt/` annotation tool are in git) |
| `*.pt` weights | `efficienttam_s.pt` and YOLO `best.pt` |
| images, videos, `.json`, `.csv`, `.parquet` | Frames, results and annotation files |
| `.venv/` | Local Python environment |

## Setup

```bash
cd etam-memory-bank-gmaw
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build that matches your CUDA version
first (see the comment in `requirements.txt`).

### Model weights

Download the weights from the project's Google Drive folder (link in
[etam-memory-bank-gmaw/README.md](etam-memory-bank-gmaw/README.md)) and place them as follows:

| File | Location |
|---|---|
| `efficienttam_s.pt` | `etam-memory-bank-gmaw/efficient_track_anything/checkpoints/` |
| `best.pt` | `etam-memory-bank-gmaw/yolov11x/` |

### Data

`tracking_data/Tracking_Set_A` and `tracking_data/Tracking_Set_B` contain the
annotation instructions (`README.md`) and the VGG Visual Tracker tool
(`vvt/`, with one empty project per clip). The clips (`clips/*.mp4`,
`clips_tracka/`), `assignment.csv` and `README_examples.png` are not in git:
copy them into the same folders locally before annotating.

## Run

Set `data.video_dir` in `etam-memory-bank-gmaw/configs/pipeline.yaml` to a
folder of `.jpg`/`.png` frames, then:

```bash
cd etam-memory-bank-gmaw
python -m src.main --cfg configs/pipeline.yaml
```

Results are written to `outputs/<timestamp>/final/`: cleaned tracks, droplet
counts (summary and per frame), overlay images, and an output video.

For the YOLO-assisted annotation tracker, paper-strict mode and evaluation, see
the docs inside `etam-memory-bank-gmaw/`:

- [README.md](etam-memory-bank-gmaw/README.md): usage details
- [SETUP_AND_INSTALLATION.md](etam-memory-bank-gmaw/SETUP_AND_INSTALLATION.md)
- [ARCHITECTURE.md](etam-memory-bank-gmaw/ARCHITECTURE.md)
- [REPRODUCTION_GUIDE.md](etam-memory-bank-gmaw/REPRODUCTION_GUIDE.md)

## References

- Bondi, E. et al. *CountVid: Open-World Object Counting in Videos.* arXiv:2506.15368
- Xiong, Y. et al. *Efficient Track Anything.* arXiv:2411.18933
- Amini, N. et al. *CountGD: Multi-Modal Open-World Counting.* arXiv:2408.00714
