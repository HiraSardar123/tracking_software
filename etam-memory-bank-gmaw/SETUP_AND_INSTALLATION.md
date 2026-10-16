# Setup & Installation Guide

Quick reference for setting up the HS-WeldCount environment.

---

python tools\annotation_tracker_server.py

## Step 1: Environment Setup

### 1.1 Create Virtual Environment (Recommended)

**Windows PowerShell**:
```powershell
# Navigate to project root
cd C:\Users\Kashif\ computers\etam-memory-bank-gmaw

# Create venv
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1
```

**Windows CMD**:
```cmd
cd C:\Users\Kashif\ computers\etam-memory-bank-gmaw
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac**:
```bash
cd c:\Users\Kashif\ computers\etam-memory-bank-gmaw
python -m venv venv
source venv/bin/activate
```

**Verify**:
```bash
python --version  # Should show 3.9+
which python      # Should point to venv/bin/python
```

---

## Step 2: Install Python Dependencies

### 2.1 Install Base Requirements

```bash
pip install --upgrade pip setuptools wheel

pip install -r requirements.txt
```

This installs:
- `torch` (CPU-only by default; override below)
- `ultralytics` (YOLOv11)
- `numpy`, `pandas`, `pyarrow`
- `pillow`, `opencv-python`, `imageio`
- `pyyaml`, `tqdm`

### 2.2 Install GPU-Accelerated PyTorch (Recommended)

Check your CUDA version:

```bash
nvidia-smi
# Output shows CUDA Version: X.Y
```

Install matching torch wheel:

**CUDA 12.1** (most common):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 12.4**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**CUDA 11.8**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only** (if no GPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2.3 Verify Installation

```python
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Expected output if GPU available:
```
True
1
```

Expected output if CPU only:
```
False
0
```

---

## Step 3: Download Model Weights

### 3.1 Required Files

Two model files are **required**:

| File | Source | Destination | Size |
|------|--------|-------------|------|
| `best.pt` | Google Drive | `yolov11x/best.pt` | ~250 MB |
| `efficienttam_s.pt` | Google Drive | `efficient_track_anything/checkpoints/efficienttam_s.pt` | ~260 MB |

### 3.2 Google Drive Download

**Link**: https://drive.google.com/drive/folders/1iY52wo8ZthAMKxm3gh3Hr5jdWVaijZyH?usp=sharing

**Steps**:

1. Open the Google Drive link in a browser
2. Right-click on `best.pt` → "Download"
3. Wait for download (~2-3 min on good connection)
4. Repeat for `efficienttam_s.pt`

### 3.3 Place Files

Create destination folders:

```bash
mkdir -p yolov11x
mkdir -p efficient_track_anything/checkpoints
```

Move downloaded files:

**Windows**:
```powershell
Move-Item ~\Downloads\best.pt yolov11x\
Move-Item ~\Downloads\efficienttam_s.pt efficient_track_anything\checkpoints\
```

**Linux/Mac**:
```bash
mv ~/Downloads/best.pt yolov11x/
mv ~/Downloads/efficienttam_s.pt efficient_track_anything/checkpoints/
```

### 3.4 Verify

```bash
ls -lh yolov11x/best.pt
ls -lh efficient_track_anything/checkpoints/efficienttam_s.pt

# Should show files of several hundred MB
```

---

## Step 4: Test Installation

### 4.1 Import Check

```bash
python -c "
from src.models import YOLODetector, EfficientTAMTracker
from src.pipeline import run_pipeline
print('✓ All imports successful')
"
```

### 4.2 Model Load Test

```bash
python -c "
from src.models import YOLODetector
yolo = YOLODetector('yolov11x/best.pt', conf=0.15)
print('✓ YOLO loaded successfully')
"
```

**Expected**: No errors; loading takes ~10 sec as model is compiled.

### 4.3 Sample Data

To test without having your own video:

```bash
# Create a tiny test video (10 frames)
mkdir -p test_frames

# Generate dummy frames
python -c "
from PIL import Image
import numpy as np

for i in range(10):
    img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    img.save(f'test_frames/frame_{i:04d}.jpg')
"

echo 'Created test_frames/ with 10 dummy frames'
```

---

## Step 5: Configuration

### 5.1 Edit `configs/pipeline.yaml`

Open in your editor:

```bash
nano configs/pipeline.yaml  # or use VS Code, etc.
```

**Essential edits**:

```yaml
data:
  video_dir: "/path/to/your/frames"   # ← Update this
  output_root: "./outputs"

yolo:
  weights: "./yolov11x/best.pt"
  conf: 0.15

efficienttam:
  ckpt: "efficient_track_anything/checkpoints/efficienttam_s.pt"
```

**For testing** (single GPU, small chunk):
```yaml
parallel:
  gpus: "auto"
  max_workers_per_gpu: 1
  strategy: ""              # Disable chunking for small tests
  frames_per_part: 1000

run:
  device: "auto"
```

### 5.2 Optional: Paper-Strict Mode

To exactly reproduce the paper (no extras):

```yaml
postprocess:
  enabled: false

output:
  render_overlays: false
  make_mp4: false
```

---

## Step 6: Run Test

### 6.1 Single Frame Test

```bash
python -c "
from PIL import Image
from src.models import YOLODetector
import numpy as np

# Create dummy frame
img = Image.new('RGB', (512, 512), color='blue')
img.save('test_frame.jpg')

# Test YOLO
yolo = YOLODetector('yolov11x/best.pt', conf=0.15)
boxes, classes, confs = yolo.detect_xyxy('test_frame.jpg')

print(f'✓ YOLO test passed. Found {len(boxes)} detections.')
"
```

### 6.2 Small Pipeline Test

```bash
# If you set up test_frames/ above:
# Edit configs/pipeline.yaml:
#   data.video_dir: "./test_frames"

python -m src.main --cfg configs/pipeline.yaml --frame-start 0 --frame-end 3

# Should complete in ~2 minutes
# Check outputs/<timestamp>/final/
```

---

## Troubleshooting Installation

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Cause**: PyTorch not installed or wrong environment

**Fix**:
```bash
# Verify environment is activated
which python  # Should show venv path

# Reinstall torch
pip install torch
```

---

### Issue: `ModuleNotFoundError: No module named 'efficient_track_anything'`

**Cause**: Package not in Python path

**Fix** (Option 1):
```bash
# From repo root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m src.main --cfg configs/pipeline.yaml
```

**Fix** (Option 2):
```bash
# Editable install
pip install -e .
```

---

### Issue: `CUDA out of memory` when running pipeline

**Solutions** (in order of preference):

1. **Reduce batch size**:
   ```yaml
   parallel:
     frames_per_part: 500  # Smaller chunks
     max_workers_per_gpu: 1
   ```

2. **Use smaller model**:
   ```yaml
   efficienttam:
     cfg: "configs/efficienttam/efficienttam_ti.yaml"  # Tiny model
   ```

3. **Use CPU** (very slow):
   ```yaml
   run:
     device: "cpu"
   ```

4. **Reduce frame resolution** (downsample input):
   ```yaml
   run:
     downsample: 2  # Use every 2nd frame
   ```

---

### Issue: `torch.cuda.is_available()` returns `False`

**Cause**: GPU not detected; CUDA driver/toolkit issue

**Diagnostics**:
```bash
nvidia-smi                    # Should show GPU list
echo $PATH                    # Should include CUDA paths
python -c "import torch; print(torch.version.cuda)"
```

**Fix**:
1. Install NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Install cuDNN: https://developer.nvidia.com/cudnn
3. Restart terminal/IDE
4. Reinstall torch with matching CUDA version (see Step 2.2)

---

### Issue: Download interrupted; partial weight file

**Cause**: Network timeout

**Fix**:
```bash
# Delete incomplete file
rm yolov11x/best.pt

# Re-download from Google Drive
# Or use alternative method below
```

---

### Issue: Long download time or "Permission denied" on Google Drive

**Alternative: Download via command-line**

If you have `gdown` installed:
```bash
pip install gdown

gdown "FOLDER_ID_HERE" --folder
```

Otherwise, check the README for alternative download links.

---

## Step 7: Verify Full Setup

Run this checklist:

```bash
#!/bin/bash
# setup_check.sh

echo "[1/5] Checking Python version..."
python --version | grep -q "3\.[9-9]" && echo "✓ Python 3.9+" || echo "✗ Update Python"

echo "[2/5] Checking CUDA..."
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

echo "[3/5] Checking model weights..."
[ -f yolov11x/best.pt ] && echo "✓ YOLO weights found" || echo "✗ Missing yolov11x/best.pt"
[ -f efficient_track_anything/checkpoints/efficienttam_s.pt ] && echo "✓ EfficientTAM weights found" || echo "✗ Missing efficienttam_s.pt"

echo "[4/5] Checking imports..."
python -c "
from src.models import YOLODetector, EfficientTAMTracker
from src.pipeline import run_pipeline
print('✓ All imports successful')
"

echo "[5/5] Checking config..."
[ -f configs/pipeline.yaml ] && echo "✓ Config found" || echo "✗ Missing configs/pipeline.yaml"

echo ""
echo "Setup check complete!"
```

Save as `setup_check.sh`, run with:
```bash
bash setup_check.sh
```

---

## Environment Variables (Optional)

### For Development

```bash
# Disable torch compilation (faster iteration)
export TORCH_COMPILE_DISABLE=1

# Enable debug logging
export DEBUG=1

# Limit to 1 GPU for debugging
export CUDA_VISIBLE_DEVICES=0
```

### For Performance Tuning

```bash
# Use all GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Disable cuDNN benchmarking (more memory, same speed)
export CUDNN_BENCHMARK=0
```

---

## Docker Setup (Optional)

If you want containerized environment:

**Dockerfile**:
```dockerfile
FROM pytorch/pytorch:2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "-m", "src.main"]
```

**Build & Run**:
```bash
docker build -t etam-weldcount .
docker run --gpus all -v /path/to/frames:/data etam-weldcount --cfg configs/pipeline.yaml
```

---

## Quick Reference: Common Commands

```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
./venv/Scripts/Activate.ps1  # Windows PowerShell

# Run full pipeline
python -m src.main --cfg configs/pipeline.yaml

# Run with 2 GPUs
# (Edit pipeline.yaml first: parallel.gpus: [0, 1])
python -m src.main --cfg configs/pipeline.yaml

# Run subset of frames
python -m src.main --cfg configs/pipeline.yaml --frame-start 0 --frame-end 100

# Disable chunking (single GPU mode)
python -m src.main --cfg configs/pipeline.yaml --no-chunks

# Check latest results
ls -la outputs/*/final/

# View generated video
ffplay outputs/*/video_out.mp4
```

---

## Next Steps

1. ✓ Install dependencies
2. ✓ Download weights
3. ✓ Update config
4. → **Read REPRODUCTION_GUIDE.md** for detailed pipeline walkthrough
5. → **Read ARCHITECTURE.md** for technical deep-dive
6. → Run on your data

---

**Support**: See REPRODUCTION_GUIDE.md section "Troubleshooting" for common issues.

**Last Updated**: August 2026
