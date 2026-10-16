# HS-WeldCount: Introduction, Problem Statement & Methodology

---

## 1. Introduction

Gas Metal Arc Welding (GMAW) is a widely used industrial process for joining metals, particularly in automotive manufacturing and structural steel construction. During GMAW, molten metal droplets are transferred from the electrode to the weld pool at rates exceeding 100 droplets per second in high-speed applications. Understanding and optimizing this droplet transfer process is critical for weld quality, process stability, and defect prevention (Bird et al., 2019; Vilarinho et al., 2021).

Traditionally, the quality and characteristics of the GMAW process have been assessed through indirect measurements such as current and voltage monitoring, or post-weld inspection of the final bead geometry. However, these methods provide limited insight into the transient droplet behavior and temporal dynamics that occur during welding. Direct observation through high-speed video has emerged as a powerful diagnostic tool, enabling frame-by-frame analysis of droplet formation, transfer, and pool interaction at millisecond timescales.

Despite the availability of high-speed video technology (capable of capturing 1,000–20,000 frames per second), manual analysis of welding videos remains time-consuming and subject to human error. A typical GMAW process video lasting just 10 seconds at 10,000 fps yields 100,000 frames. Manually counting droplets, tracking their trajectories, and quantifying transfer characteristics across such datasets is impractical. Consequently, most high-speed welding studies analyze only small subsets of data, limiting the statistical robustness of conclusions.

Recent advances in computer vision and deep learning have demonstrated remarkable progress in object detection, instance segmentation, and multi-object tracking across diverse domains (He et al., 2017; Dosovitskiy et al., 2020; Kirillov et al., 2022). Models such as YOLO (Ultralytics, 2023) excel at per-frame detection, while foundation models like Segment Anything (SAM) (Kirillov et al., 2023) and its efficient variants enable precise, prompt-guided segmentation. Building on these advances, we present **HS-WeldCount**, an automated pipeline for detecting, segmenting, tracking, and counting droplets in high-speed GMAW video with minimal manual intervention.

The key contributions of this work are:

1. **End-to-end automated droplet detection and counting** in high-speed welding video, eliminating manual frame-by-frame inspection.

2. **A three-stage pipeline** combining per-frame detection (YOLO v11), efficient segmentation (EfficientTAM), and long-term tracking with temporal filtering to handle occlusions and per-frame noise.

3. **Scalable multi-GPU architecture** for processing large video datasets, enabling practical deployment on industrial high-speed video streams.

4. **Comprehensive evaluation** on real GMAW welding videos, demonstrating accuracy and robustness across varying process conditions.

This work bridges the gap between state-of-the-art computer vision and domain-specific challenges in welding process monitoring, offering a practical tool for researchers and manufacturers to accelerate data-driven optimization of GMAW processes.

---

## 2. Problem Statement

### 2.1 Challenges in High-Speed Welding Video Analysis

High-speed GMAW video analysis presents several technical and practical challenges that make automated droplet counting non-trivial:

**Challenge 1: High Frame Volume**
- Typical high-speed welding capture rates: 5,000–20,000 fps
- A 10-second weld sequence yields 50,000–200,000 frames
- Manual inspection: ~10–30 seconds per frame → 140–200 hours for a single 10-second sequence
- Impractical for iterative process optimization or large-scale studies

**Challenge 2: Weak Visual Contrast**
- Droplets exhibit low contrast against the bright weld pool and arc plasma
- Variable illumination due to thermal radiation during the arc
- Droplets often appear semi-transparent or with similar color to surroundings
- Standard bounding-box detectors (trained on everyday objects) fail without domain-specific fine-tuning

**Challenge 3: Occlusion and Overlap**
- Multiple droplets frequently overlap in the image plane
- Occlusion by weld pool, nozzle, or shielding gas
- Per-frame detection models detect multiple overlapping bounding boxes for single objects
- Segmentation required to separate overlapping regions

**Challenge 4: Scale Variation**
- Droplet size varies with transfer mode (spray transfer: 2–5 mm; globular: 5–8 mm)
- In-plane droplet position varies from electrode to pool (distance: 10–40 mm)
- Perspective distortion and out-of-plane motion
- Requires multi-scale detection and segmentation

**Challenge 5: Temporal Noise and Artifacts**
- Per-frame segmentation from independent models produces inconsistent masks across frames
- A single droplet may yield multiple overlapping masks per frame
- Gaps in tracks due to occlusion or failed detection
- Requires temporal consistency filtering

**Challenge 6: Limited Labeled Data**
- Publicly available welding video datasets are rare
- Manual annotation of high-speed video is labor-intensive
- Domain-specific fine-tuning of general vision models is often necessary

### 2.2 Existing Approaches and Limitations

**Manual Counting**:
- Accurate but impractical for large volumes
- Subject to fatigue and human error
- Cost-prohibitive for routine analysis

**Classical Computer Vision (Thresholding, Contour Detection)**:
- Require carefully tuned parameters for each lighting condition
- Fail under variable illumination
- Cannot handle occlusion or complex backgrounds
- Low accuracy on complex scenes

**General-Purpose Object Detection (YOLO, Faster R-CNN)**:
- YOLO provides fast per-frame detection but:
  - Bounding boxes alone insufficient for precise droplet localization
  - Multiple boxes per droplet (no merging mechanism)
  - Cannot separate overlapping droplets
- Fine-tuning requires domain-specific labeled data

**Optical Flow and Classical Tracking**:
- Motion-based tracking fails when droplets move slowly or are stationary
- Cannot recover from occlusions
- Unreliable in high-clutter environments

### 2.3 Research Question

**How can we automatically detect, precisely segment, robustly track, and accurately count droplets in high-speed GMAW video without extensive manual labeling, while handling occlusions, scale variation, temporal noise, and real-time processing constraints?**

---

## 3. Methodology

### 3.1 System Overview

We propose a **three-stage pipeline** that combines state-of-the-art vision models with domain-specific temporal filtering:

```
Input Video (N frames)
         ↓
[STAGE 1] Per-Frame Detection + Segmentation
   ├─ YOLO v11x: Bounding box detection (droplets + weld pool)
   └─ EfficientTAM: Prompt-guided instance segmentation
   ↓ Output: Per-frame masks M_f = {mask_0, mask_1, ..., mask_K}
         ↓
[STAGE 2] Temporal Deduplication Filter
   └─ Forward IoU matching within local temporal windows
   ↓ Output: Deduplicated per-frame masks
         ↓
[STAGE 3] Long-Term Tracking + Discovery
   └─ Greedy IoU-based global ID assignment across entire sequence
   ↓ Output: Global tracks T = {Track_1, Track_2, ..., Track_M}
         ↓
[POST-PROCESS] Optional Temporal & Domain Filters (not in core methodology)
   ├─ Temporal track smoothing
   ├─ Physics-based filtering (falling droplets)
   └─ Domain-specific heuristics (droplet lifetime, size, motion)
         ↓
Output: Droplet count, tracks, visualizations
```

Each stage is designed to be modular, allowing independent optimization and validation.

---

### 3.2 Stage 1: Per-Frame Detection and Segmentation

#### 3.2.1 Detection: YOLOv11x

We employ YOLOv11x (Ultralytics, 2023), a state-of-the-art real-time object detector, for frame-by-frame droplet and weld-pool detection.

**Training Data**: 
- Custom-annotated GMAW video frames
- Two classes: Class 0 (weld pool), Class 1 (droplets)
- Class 0 masks ignored in final counting; used for visualization

**Detection Parameters**:
- Confidence threshold: 0.15 (optimized for recall; false positives handled by segmentation)
- IoU threshold (NMS): 0.30 (allows multiple overlapping detections per object)
- Input: RGB frame; Output: Bounding boxes (x₁, y₁, x₂, y₂) with class labels

**Rationale**: 
- YOLO provides fast, coarse localization (bounding boxes)
- Conservative NMS avoids suppressing overlapping droplets
- Confidences threshold prioritizes recall over precision; segmentation refines later

#### 3.2.2 Segmentation: EfficientTAM

Bounding boxes are passed as prompts to EfficientTAM (Xiong et al., 2024), a lightweight variant of the Segment Anything Model (SAM).

**Architecture**:
- **Image Encoder**: ViT-base backbone for frame encoding
- **Prompt Encoder**: Converts bounding box coordinates into embedding
- **Mask Decoder**: Produces per-box binary segmentation mask
- **Memory Module**: Temporal context aggregation across frames

**Key Features**:
- **Prompt-guided segmentation**: Bounding box → precise pixel-level mask
- **Efficient design**: ~260 MB checkpoint; ~50–200 ms per frame inference
- **Temporal memory**: Maintains latent state across frames, enabling consistency

**Inference**:
```
For each frame f:
  1. Load image I_f
  2. Run YOLO → boxes B_f = {box_0, box_1, ..., box_n}
  3. For each box in B_f:
       a. Encode prompt (box coordinates)
       b. Feed to EfficientTAM along with I_f
       c. Output: Binary mask M_{f,i}
  4. Collect all masks: M_f = {M_{f,0}, M_{f,1}, ..., M_{f,n}}
```

**Output**: Per-frame instance masks (pixel-level), one per detected object.

---

### 3.3 Stage 2: Temporal Deduplication Filter

#### 3.3.1 Problem: Per-Frame Redundancy

YOLO + EfficientTAM can produce multiple overlapping masks per physical droplet in a single frame. This occurs when:
- A single droplet triggers multiple YOLO boxes (e.g., due to aspect-ratio diversity)
- Partial occlusions result in multiple prompt responses
- Conservative NMS retains overlapping detections

**Example**: Droplet A may appear as masks {M₁, M₂, M₃} in frame f, with IoU(M₁, M₂) > 0.9, but only one represents the true object.

#### 3.3.2 Solution: Forward IoU Matching

We apply a lightweight temporal filter to remove near-duplicates within a local window:

**Algorithm**:
```
Input: Per-frame masks M = {M_0, M_1, ..., M_{T-1}}
       Window size w (typically 2–4 frames)
       IoU threshold τ (typically 0.05–0.10)

Output: Deduplicated masks M' ⊂ M

For each frame f:
  For each mask m in M_f:
    is_duplicate ← FALSE
    For each future frame g in [f+1, min(f+w, T-1)]:
      For each mask m' in M_g:
        if IoU(m, m') > τ:
          is_duplicate ← TRUE
          break
    if NOT is_duplicate:
      Retain m in M'_f
```

**Rationale**:
- **Forward-only matching**: More efficient than bidirectional; avoids back-dependencies
- **Soft IoU threshold**: Only high-overlap masks are marked as duplicates
- **Local window**: Balances robustness (looking ahead) with efficiency

**Output**: Deduplicated per-frame masks M', expected to have fewer artifacts.

---

### 3.4 Stage 3: Long-Term Tracking and Identity Assignment

#### 3.4.1 Global Trajectory Consistency

After deduplication, we assign global track IDs (T₁, T₂, ..., T_M) to connect masks across the entire video sequence. This solves:
- **Track initialization**: First appearance of a droplet
- **Track linking**: Connecting a mask in frame f to the same droplet in frame f+1
- **Track termination**: Last frame of a droplet
- **New object discovery**: Objects appearing after occlusion or entering the frame

#### 3.4.2 Greedy IoU Matching

We employ a frame-by-frame greedy matching strategy:

**Algorithm**:
```
Input: Deduplicated per-frame masks M' = {M'_0, M'_1, ..., M'_{T-1}}
       IoU overlap threshold θ (typically 0.10)
       Maximum gap max_skip (typically 0 frames)

Output: Global tracks T = {Track_1, Track_2, ..., Track_M}
        where each Track_i = {(frame_i, mask_i), (frame_j, mask_j), ...}

Initialize:
  T ← {} (empty track dictionary)
  active_tracks ← {} (current track states)
  next_id ← 1

For each frame f in [0, T-1]:
  matched_masks ← ∅
  
  For each active track τ:
    last_frame ← frame of most recent mask in τ
    gap ← f - last_frame
    
    if gap > max_skip:
      Remove τ from active_tracks
      continue
    
    prev_mask ← most recent mask of τ
    
    best_match ← NULL
    best_iou ← -1
    
    For each unmatched mask m in M'_f:
      iou ← IoU(prev_mask, m)
      if iou > best_iou:
        best_iou ← iou
        best_match ← m
    
    if best_iou > θ:
      Append (f, best_match) to τ
      active_tracks[τ] ← f
      matched_masks ← matched_masks ∪ {best_match}
  
  For each unmatched mask m in M'_f:
    Create new track: T[next_id] ← [(f, m)]
    active_tracks[next_id] ← f
    next_id ← next_id + 1

Return T
```

**Key Design Decisions**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **θ (overlap threshold)** | 0.10 | 10% IoU suffices to link consecutive frames; balances over-linking and under-linking |
| **max_skip** | 0–6 | Allow gaps for temporary occlusions; 0 = strict (no gaps); 6 = permissive (0.3 sec at 20 kfps) |
| **Greedy matching** | Best-match per track | Computationally efficient; avoid global assignment problem |

#### 3.4.3 Handling Scale and Occlusion

- **Scale variation**: IoU is scale-invariant; small and large droplets tracked equally
- **Occlusion**: If a droplet is occluded (no mask detected), the track is paused; if re-detected after gap ≤ max_skip, linking resumes
- **Multiple occlusions**: Tracks that remain absent for > max_skip frames are considered terminated; re-appearance starts a new track ID

**Output**: Global track assignments with consistent IDs across frames.

---

### 3.5 Scalability: Multi-GPU Chunked Processing

For large videos (thousands of frames), processing entire sequences on a single GPU is memory-prohibitive. We implement a chunked, multi-GPU strategy:

#### 3.5.1 Chunking Strategy

```
Input:  Video with N frames
        K available GPUs
        Chunk size c (e.g., 1000 frames)
        Overlap size o (e.g., 3 frames)

Split into overlapping chunks:
  Chunk 1: frames [0, c]           → GPU 0
  Chunk 2: frames [c-o, 2c-o]      → GPU 1 (overlap: [c-o, c])
  Chunk 3: frames [2c-2o, 3c-2o]   → GPU 2 (overlap: [2c-2o, 2c-o])
  ...

Each GPU runs Stages 1–3 independently on its chunk.
```

#### 3.5.2 ID Stitching via Overlap Frame Re-identification

After chunked processing, per-chunk track IDs are local (e.g., Track_0, Track_1 in each chunk). We merge into global IDs by re-identifying masks in overlap frames:

```
For each pair of adjacent chunks C_i and C_{i+1}:
  overlap_frames ← frames in both chunks
  
  For each track τ_i in C_i (local ID):
    For each track τ_{i+1} in C_{i+1} (local ID):
      iou_score ← mean IoU across overlap frames
      
      if iou_score > threshold:
        τ_i and τ_{i+1} represent same droplet
        Merge: rename τ_{i+1} IDs to τ_i IDs in C_{i+1}

Append all chunks into single global track dictionary.
```

**Advantage**: Near-linear speedup with K GPUs; only overlap frames recomputed.

---

### 3.6 Post-Processing (Optional, Not in Core Methodology)

While the three-stage pipeline constitutes the core methodology, we optionally apply domain-specific filters:

#### 3.6.1 Temporal Track Filtering

- **Minimum track length**: Remove tracks lasting < L frames (e.g., L = 6), likely noise
- **Yoyo filtering**: Discard brief disconnections (track vanishes then reappears within few frames)

#### 3.6.2 Physics-Based Filtering

- **Falling droplet constraint**: Droplets fall (Δy > 0) due to gravity; flag upward motion as anomaly
- **Velocity bounds**: Droplets have bounded velocity; unrealistic jumps indicate errors
- **Pool containment**: Ignore objects that enter the weld pool and don't exit

#### 3.6.3 Size Filtering

- **Minimum area**: Remove masks < 2 pixels (noise)
- **Maximum area**: Cap size at physically plausible droplet diameter

**Note**: These filters are **not** applied in "paper-strict" mode to ensure reproducibility of the core three-stage methodology.

---

### 3.7 Implementation Details

#### 3.7.1 Mask Representation

Masks are stored as **coordinate lists** (y-indices, x-indices) rather than dense binary matrices:
- **Compactness**: Only foreground pixels stored
- **Efficiency**: Fast IoU computation via vectorized operations
- **Flexibility**: Supports variable-size masks without padding

**IoU Computation**:
```
IoU(mask_a, mask_b, H, W) = |mask_a ∩ mask_b| / |mask_a ∪ mask_b|

Implementation:
  1. Convert coordinate lists to binary matrices (sparse)
  2. Logical AND → intersection
  3. Logical OR → union
  4. Compute ratio
```

#### 3.7.2 Outputs

**Primary outputs** (parquet tables):
- `tracks.parquet`: Columns [track_id, frame, mask_pixels, area, bbox]
- `centroids.parquet`: Columns [track_id, frame, y_center, x_center]

**Derived outputs** (JSON/CSV):
- `droplet_count_summary.json`: Total count, per-frame statistics
- `droplet_count_per_frame.csv`: Frame-by-frame counts

**Visualization** (PNG/MP4):
- Per-frame overlays with drawn masks and track IDs
- MP4 video at user-specified frame rate (default 30 fps)

---

## 4. Summary of Methodology

| Stage | Input | Method | Output |
|-------|-------|--------|--------|
| **1** | Video frames | YOLOv11 + EfficientTAM | Per-frame instance masks |
| **2** | Per-frame masks | Forward IoU deduplication | Deduplicated per-frame masks |
| **3** | Deduplicated masks | Greedy IoU-based tracking | Global track IDs + trajectories |
| **Post** | Tracks | Optional physics/temporal filters | Cleaned tracks (optional) |

The pipeline is:
- **Modular**: Each stage can be developed and validated independently
- **Efficient**: Stages 2 & 3 are O(N) to O(N²) in video length; Stage 1 is bottleneck
- **Scalable**: Multi-GPU support via chunking with overlap-based stitching
- **Reproducible**: Deterministic algorithms (no randomness); all parameters configurable

---

## References

Bondi, E., et al. (2025). "CountVid: Open-World Object Counting in Videos." *arXiv:2506.15368*.

Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICCV*.

He, K., et al. (2017). "Mask R-CNN." *ICCV*.

Kirillov, A., et al. (2023). "Segment Anything." *ICCV 2023*.

Ultralytics. (2023). "YOLOv11 Documentation." https://docs.ultralytics.com/models/yolov11/

Xiong, Y., et al. (2024). "Efficient Track Anything." *arXiv:2411.18933*.

---

**Word Count**: ~2,500  
**Status**: Ready for integration into academic paper or technical report
