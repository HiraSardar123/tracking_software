# Tracking Software: GMAW Droplet Tracking, Counting and Annotation

This repository contains everything needed to **automatically track and count
metal droplets** in high-speed gas metal arc welding (GMAW) video, and to
**manually annotate droplet tracks** in a browser to create ground truth.

It has two top-level folders:

| Folder | What it is | How you use it |
|---|---|---|
| [`etam-memory-bank-gmaw/`](etam-memory-bank-gmaw/) | **The tracking software** (Python). Detects droplets with YOLOv11x, segments and tracks them with EfficientTAM, and counts them. It also contains a small local server that adds YOLO-assisted tracking to the annotation tool. | Run from a terminal |
| [`tracking_data/`](tracking_data/) | **The annotation tool and datasets.** Two annotation sets (`Tracking_Set_A`, `Tracking_Set_B`), each with instructions and its own copy of the VGG Visual Tracker (VVT) browser tool. | Open `vvt/index.html` in a browser |

> **The videos, CSVs, images and model weights are not in this repository.**
> They are private or too large for GitHub. After cloning, you must copy them in
> yourself (see [Step 2](#step-2-add-the-files-that-are-not-in-git)).

---

## Contents

1. [Folder structure](#folder-structure)
2. [Step 1: Download the repository](#step-1-download-the-repository)
3. [Step 2: Add the files that are not in git](#step-2-add-the-files-that-are-not-in-git)
4. [Step 3: Install the Python environment](#step-3-install-the-python-environment)
5. [Workflow A: Annotate clips in the browser (VVT tool)](#workflow-a-annotate-clips-in-the-browser-vvt-tool)
6. [Workflow B: Run automatic tracking and counting](#workflow-b-run-automatic-tracking-and-counting)
7. [Troubleshooting](#troubleshooting)
8. [What is and is not tracked by git](#what-is-and-is-not-tracked-by-git)
9. [References](#references)

---

## Folder structure

```
tracking_software/
├── README.md                         <- this file
├── .gitignore                        <- keeps videos, CSVs, images, weights out of git
│
├── etam-memory-bank-gmaw/            <- TRACKING SOFTWARE (Python)
│   ├── src/                          <- pipeline: YOLO -> EfficientTAM -> tracking
│   │   └── main.py                   <- entry point: python -m src.main
│   ├── tools/
│   │   ├── annotation_tracker_server.py  <- YOLO helper for the VVT tool (Set B)
│   │   ├── count_droplets.py, postprocess_tracks_min.py, render_overlays_min.py
│   │   ├── benchmark_eval.py, make_intervals.py
│   ├── configs/pipeline.yaml         <- main settings (input frames folder, thresholds)
│   ├── efficient_track_anything/     <- EfficientTAM model code
│   │   └── checkpoints/              <- put efficienttam_s.pt here   (not in git)
│   ├── yolov11x/                     <- put best.pt here             (not in git)
│   ├── requirements.txt
│   ├── file2mp4.py                   <- turn a folder of frames into an .mp4
│   └── README.md, SETUP_AND_INSTALLATION.md, ARCHITECTURE.md, REPRODUCTION_GUIDE.md
│
└── tracking_data/                    <- ANNOTATION TOOL + DATA
    ├── Tracking_Set_A/
    │   ├── README.md                 <- full annotation instructions (read this!)
    │   ├── assignment.csv            <- clips assigned to you            (not in git)
    │   ├── README_examples.png       <- example image for the README     (not in git)
    │   ├── clips/clip_XXX.mp4        <- the video clips                  (not in git)
    │   ├── annotations/              <- put your finished annotations here
    │   └── vvt/
    │       ├── index.html            <- THE ANNOTATION TOOL: open in Chrome/Firefox
    │       └── projects/clip_XXX.json  <- one empty project per clip
    └── Tracking_Set_B/              <- index.html file in this folder is changed to work with yolo.
        ├── (same layout as Set A)
        ├── clips_tracka/             <- extra clip set, if you were given it (not in git)
        └── vvt/index.html            <- same tool + optional YOLOv11x tracker
```

**Set A and Set B use different clips.** Each set's `vvt/projects/` holds an
empty project for each of its clips. The two tools work the same way, except
that **only Set B's tool** has the extra **YOLOv11x** tracker choice (see
[A4](#a4-optional-yolo-assisted-tracking-set-b-only)).

---

## Step 1: Download the repository

```bash
git clone https://github.com/HiraSardar123/tracking_software.git
cd tracking_software
```

(Or on GitHub: **Code → Download ZIP**, then unzip it.)

---

## Step 2: Add the files that are not in git

Get these files from the project owner (shared drive / Google Drive) and place
them **exactly** at these paths:

### Data (needed for annotation and for running on the clips)

| File | Put it here |
|---|---|
| Set A clips | `tracking_data/Tracking_Set_A/clips/clip_XXX.mp4` |
| Set A assignment sheet | `tracking_data/Tracking_Set_A/assignment.csv` |
| Set A example image | `tracking_data/Tracking_Set_A/README_examples.png` |
| Set B clips | `tracking_data/Tracking_Set_B/clips/clip_XXX.mp4` |
| Set B extra clips (if any) | `tracking_data/Tracking_Set_B/clips_tracka/clip_XXX.mp4` |
| Set B assignment sheet | `tracking_data/Tracking_Set_B/assignment.csv` |
| Set B example image | `tracking_data/Tracking_Set_B/README_examples.png` |

Create the `clips/` and `annotations/` folders if they do not exist. Git does
not store empty folders.

### Model weights (needed for the Python pipeline and the YOLO tracker)

Request access to them.

| File | Put it here |
|---|---|
| `efficienttam_s.pt` | `etam-memory-bank-gmaw/efficient_track_anything/checkpoints/efficienttam_s.pt` |
| `best.pt` (YOLOv11x) | `etam-memory-bank-gmaw/yolov11x/best.pt` |

> If you **only** want to annotate with the plain VVT tool (no YOLO), you need
> just the clips. You can skip the weights and Step 3.

---

## Step 3: Install the Python environment

Requires **Python 3.9 or newer** (tested with 3.11). A CUDA GPU is strongly
recommended for the pipeline. The YOLO annotation helper also works on CPU.

```bash
cd etam-memory-bank-gmaw
python -m venv .venv
```

Activate it:

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```
```bash
# Linux / macOS
source .venv/bin/activate
```

Install PyTorch **first**, matching your CUDA version (check it with `nvidia-smi`):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
# pip install torch --index-url https://download.pytorch.org/whl/cu124 # CUDA 12.4
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # no GPU
```

Then the rest:

```bash
pip install -r requirements.txt
```

Check that the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Full details: [etam-memory-bank-gmaw/SETUP_AND_INSTALLATION.md](etam-memory-bank-gmaw/SETUP_AND_INSTALLATION.md).

---

## Workflow: Annotation Strategies
### For the tracking annotations, I personally used A4, in which I used the tool, with yolo assisted annotations. it is the recommended approach.
### For the data, I would recommend pushing data to the tracking_set_b, since the html file is optimized to work with yolo.
---

## Workflow A: Annotate clips in the browser (VVT tool)

This is for creating ground-truth droplet tracks by hand. **Read the full
instructions first:**
[`Tracking_Set_A/README.md`](tracking_data/Tracking_Set_A/README.md) /
[`Tracking_Set_B/README.md`](tracking_data/Tracking_Set_B/README.md)
(which droplets to annotate, start and end rules, merges, saving).

### A1. Open the tool

1. Go to `tracking_data/Tracking_Set_A/vvt/` (or `Tracking_Set_B/vvt/`).
2. Double-click **`index.html`**. It opens in Chrome or Firefox.
   It runs fully offline: no account, no install, no internet.

### A2. Load a clip

1. Click the **folder icon** in the top bar (**Open a VIA Project**) and choose
   `vvt/projects/clip_XXX.json` (a clip listed in your `assignment.csv`).
2. The page says **File Not Found!** That is normal. In the **File Source** row
   click **Choose File** and pick `clips/clip_XXX.mp4` (the same number).
3. The video appears with a timeline below it.

### A3. Annotate and save

- `Alt` + drag starts a **new** droplet track. A plain drag (no `Alt`) adds a
  box to the highlighted track.
- `→` / `←` moves one frame. `Shift+N` / `Shift+P` moves 5 frames. `Space`
  plays or pauses.
- Pressing `t` auto-tracks the box forward. Any key stops it. **Always check
  the boxes it creates.**
- Click the **disk icon** (**Save current VIA Project**). The browser downloads
  `via_project_<date>.json`. Move it to `annotations/` and rename it
  `clip_XXX_annotations.json`.

### A4. Optional: YOLO-assisted tracking (Set B only)

Set B's tool can use the trained YOLOv11x detector as a tracker. It is usually
more accurate on these droplets than VVT's built-in KCF/CFNet trackers. This
needs Step 2 (weights) and Step 3 (Python environment).

**1. Start the helper server.** Leave this terminal open while you annotate:

```bash
cd etam-memory-bank-gmaw
# activate .venv first (see Step 3)
python tools/annotation_tracker_server.py
```

You should see:

```
YOLO annotation tracker listening on http://127.0.0.1:8765
```

Options: `--port 8765`, `--conf 0.15`, `--iou 0.30`, `--weights path/to/best.pt`.
The server only listens on your own computer (`127.0.0.1`).

**2. Use it in the browser:**

1. Open `tracking_data/Tracking_Set_B/vvt/index.html` and load a clip (A2).
2. Select a box, and in the **Tracker** field choose **`YOLOv11x`**.
3. Pause the video, draw or adjust the box, and press `t`. Press any key to stop.
4. Check every generated box before saving. Tracking stops by itself when the
   detection is ambiguous.

**Split droplet:** pause on the split frame, draw the first box, then draw the
second box on the same frame **without** `Alt` (same track row selected), and
press `t`. Both pieces are tracked under the same object name.

Stop the server with `Ctrl+C` when you are done.

---

## Workflow B: Run automatic tracking and counting

The pipeline runs three stages: **YOLOv11x** detection on every frame,
**EfficientTAM** segmentation and propagation, then temporal filtering and
long-term tracking with new-droplet discovery. It ends with clean-up, counting,
overlay images and an output video.

### B1. Prepare a folder of frames

The pipeline reads a **folder of `.jpg` frames**, not an `.mp4`. If you have
raw frames from the camera, use those. To extract frames from a clip, run this
from `etam-memory-bank-gmaw/` with the environment active:

```bash
python -c "import cv2,os,sys; v=cv2.VideoCapture(sys.argv[1]); o=sys.argv[2]; os.makedirs(o,exist_ok=True); i=0
while True:
    ok,f=v.read()
    if not ok: break
    cv2.imwrite(os.path.join(o,f'{i:06d}.jpg'),f); i+=1
print(i,'frames written to',o)" ../tracking_data/Tracking_Set_B/clips/clip_001.mp4 ../frames/clip_001
```

This writes `000000.jpg`, `000001.jpg`, ... into `tracking_software/frames/clip_001/`.
That folder is git-ignored.

### B2. Point the config at the frames

Open `etam-memory-bank-gmaw/configs/pipeline.yaml` and set `video_dir` to
your frames folder. Relative paths are resolved from `etam-memory-bank-gmaw/`.

```yaml
data:
  video_dir: "../frames/clip_001"   # <- your frames folder
  output_root: "./outputs"
```

> The value in the repository is an absolute path on the original author's
> machine. **You must change it.**

Other settings you may want to change:

| Setting | Meaning |
|---|---|
| `run.device` | `auto`, `cuda` or `cpu` |
| `parallel.gpus` | GPU ids to use, e.g. `[0]` or `[0, 1]` |
| `parallel.frames_per_part` | frames per chunk (lower it if you run out of GPU memory) |
| `yolo.conf` | detection confidence threshold |
| `output.fps` | frame rate of the output video |

### B3. Run

```bash
cd etam-memory-bank-gmaw
# activate .venv first
python -m src.main --cfg configs/pipeline.yaml
```

Useful options:

```bash
python -m src.main --cfg configs/pipeline.yaml --frame-start 0 --frame-end 500   # only a range
python -m src.main --cfg configs/pipeline.yaml --no-chunks                        # single process, no chunking
```

### B4. Results

Results are written to `etam-memory-bank-gmaw/outputs/<timestamp>/final/`:

| File | Content |
|---|---|
| `droplet_count_summary.json` | total droplet count |
| `droplet_count_per_frame.csv` | count per frame |
| `tracks_clean.parquet`, `centroids_clean.parquet` | cleaned tracks and centroids |
| `overlays_post/*.png` | frames with the masks and IDs drawn on them |
| `video_out.mp4` | overlay video |

`outputs/` is git-ignored, so results never get pushed.

**Paper-strict mode** (no extra welding-specific filters or rendering): in
`pipeline.yaml` set `postprocess.enabled: false`, `output.render_overlays: false`
and `output.make_mp4: false`.

More: [etam-memory-bank-gmaw/README.md](etam-memory-bank-gmaw/README.md),
[ARCHITECTURE.md](etam-memory-bank-gmaw/ARCHITECTURE.md),
[REPRODUCTION_GUIDE.md](etam-memory-bank-gmaw/REPRODUCTION_GUIDE.md).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Frames directory not found` | `data.video_dir` in `pipeline.yaml` is wrong. See B2. |
| `No frames (.jpg) found` | The folder has no `.jpg` files. Extract frames (B1). |
| `FileNotFoundError ... best.pt` / `efficienttam_s.pt` | The weights are missing. See Step 2. |
| `torch.cuda.is_available()` is `False` | Reinstall torch with the CUDA wheel that matches `nvidia-smi`, or set `run.device: cpu`. |
| GPU out of memory | Lower `parallel.frames_per_part`, or use `--frame-start/--frame-end`. |
| VVT says **File Not Found!** | Normal. Choose the `.mp4` in **File Source** (A2). |
| `YOLOv11x` tracker does nothing | The helper server is not running, or you are in Set A's tool. Start the server (A4) and use Set B. |
| Arrow keys move a box instead of the video | A box is selected. Click an empty part of the image. |
| `Activate.ps1 cannot be loaded` (Windows) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |

---

## What is and is not tracked by git

**In git:** all Python code, `configs/*.yaml`, documentation, both
`tracking_data/*/README.md` files, and the complete `vvt/` annotation tool for
each set (HTML, JS, its small CFNet model, and the empty `projects/*.json`).

**Never pushed (see `.gitignore`):**
- videos (`*.mp4`, `*.avi`, `*.mov`, ...), images (`*.png`, `*.jpg`, ...)
- tables and annotations: `*.csv`, `*.json` (except inside `vvt/`), `*.parquet`
- `annotations/`, `clips/`, `frames/`, `outputs/` folders
- model weights (`*.pt`, `*.pth`, ...)
- Python environments (`.venv/`) and `.env` files

> **Your finished annotation files are not pushed.** `annotations/` is
> ignored. Send them to the project owner directly, as described in the set's
> README.

---

## References

- Bondi, E. et al. *CountVid: Open-World Object Counting in Videos.* [arXiv:2506.15368](https://arxiv.org/abs/2506.15368)
- Xiong, Y. et al. *Efficient Track Anything.* [arXiv:2411.18933](https://arxiv.org/abs/2411.18933)
- Amini, N. et al. *CountGD: Multi-Modal Open-World Counting.* [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)
- Sridhar, P., Dutta, A., Zisserman, A. *VGG Visual Tracker (VVT)*, built on VGG Image Annotator (VIA). https://www.robots.ox.ac.uk/~vgg/software/vvt/ (BSD 2-clause)
