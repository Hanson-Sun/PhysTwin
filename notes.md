# PhysTwin Project Notes

A central place to explore and learn about this repository.

---

## TODO and progress

What i did
1. use DA3 + camera intrinsics/extrinsics to get depth (`parse_depth_da3.py`). We actually use the new camera intrinsics from DA3 prediction, the spatial locations (extrinsics) are the same, but the intrinsics are different from the real camera intrinsics.
  - there seem to be some issues for this, the sloth isnt attached to the sloth...? maybe using different intrinsics is the problem? will need to do more investigation, but it looks promising! the depth reconstruction is pretty good.
  - also tried VDA + affine adjustment per camera (it sucks arms are not connected, a ton of issues, `infer_depth.py` + `depth_scale_align.py`)
  - tried DUST3R + projection and reprojection back to real camera view (has some math issues) `parse_depth.py`
2. wrote some scripts to visualize the depth quality (`compare_depth.py`)
3. added a 3d visualization script to see the point cloud from different cameras `visualize_3d_scene.py`
4. yeah random corrections here and there to make everything work (align depth and color resolution)


What to do next
1. fix the depth alignment issues, make sure the point clouds from different cameras align well (maybe match the intrinsics?), will check again
2. look into the physics reconstruction part, begin to understand new method and see if i can integrate it with the pipeline.

## Project Overview

**PhysTwin** is a framework for physics-informed reconstruction and simulation of deformable objects from videos. It creates interactive digital twins of real-world deformable objects (cloth, stuffed toys, ropes) that can be manipulated in real-time.

---

## High-Level Pipeline Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  1. DATA        │ -> │  2. DATA         │ -> │  3. PHYSICS     │ -> │  4. INTERACTIVE │
│  COLLECTION     │    │  PROCESSING      │    │  OPTIMIZATION   │    │  PLAYGROUND     │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
   RGB-D Cameras          Segmentation           Zero-order +           Keyboard control
   + Calibration          Tracking               First-order            + Visualization
                          3D Lifting             Optimization
```

---

## Step 1: Raw Data Collection

You need **multi-view RGB-D cameras** (e.g., Intel RealSense). For each case, create a folder under `data/different_types/{case_name}/` with:

| File/Folder | Description |
|-------------|-------------|
| `color/` | RGB images per camera: `color/0/0.png`, `color/0/1.png`, ... and `color/0.mp4`, `color/1.mp4`, ... |
| `depth/` | Depth images per camera: `depth/0/0.png`, `depth/0/1.png`, ... |
| `calibrate.pkl` | List of 4×4 camera-to-world transformation matrices (one per camera) |
| `metadata.json` | Camera intrinsics, serial numbers, FPS, resolution (WH), frame count |

### Example `metadata.json` structure:
```json
{
  "intrinsics": [
    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    ...
  ],
  "serial_numbers": ["cam1", "cam2", "cam3"],
  "fps": 30,
  "WH": [848, 480],
  "frame_num": 62
}
```

### Example `calibrate.pkl` structure:
- Python list of 3 numpy arrays, each 4×4 (camera-to-world matrices)

---

## Step 2: Add Your Case to Configuration

Add a row to `data_config.csv`:
```
your_case_name,category,shape_prior
```

- **category**: Object type for segmentation (e.g., `cloth`, `toy`, `sloth`, `twine`, `package`)
- **shape_prior**: `True` for 3D objects (uses Trellis for shape reconstruction), `False` for flat objects like cloth

### Existing cases in the repo:
| Case Name | Category | Shape Prior |
|-----------|----------|-------------|
| rope_double_hand | twine | True |
| double_stretch_sloth | sloth | True |
| single_lift_cloth_1 | cloth | False |
| double_lift_cloth_3 | cloth | False |
| ... | ... | ... |

---

## Step 3: Data Processing

### Single case:
```bash
python process_data.py --base_path ./data/different_types --case_name your_case_name --category your_category [--shape_prior]
```

### Batch process all cases:
```bash
python script_process_data.py
```

### Processing Pipeline Steps:

| Step | Script | Output | Description |
|------|--------|--------|-------------|
| 1. Segmentation | `data_process/segment.py` | `mask/` | Object and hand masks using Grounded-SAM2 |
| 2. Image Upscale | `data_process/image_upscale.py` | `shape/high_resolution.png` | High-res image for Trellis |
| 3. Shape Prior | `data_process/shape_prior.py` | `shape/` | 3D mesh from Trellis (if enabled) |
| 4. Dense Tracking | `data_process/dense_track.py` | `cotracker/` | 2D point tracking using Co-Tracker |
| 5. Lift to 3D | `data_process/data_process_pcd.py` | `pcd/` | Point clouds in world coordinates |
| 6. Mask Processing | `data_process/data_process_mask.py` | cleaned masks | Filter noise from masks |
| 7. Data Tracking | `data_process/data_process_track.py` | `track_process_data.pkl` | 3D tracking data |
| 8. Alignment | `data_process/align.py` | aligned shape | Align shape prior with observations |
| 9. Final Data | `data_process/data_process_sample.py` | `final_data.pkl` | Ready for physics training |

### After processing, also run:
```bash
python export_gaussian_data.py      # Export data for Gaussian training
python export_video_human_mask.py   # Export human masks for visualization
```

---

## Step 4: Physics Training (Build the PhysTwin)

### 4a. Zero-Order Optimization (CMA-ES)
~12 minutes per case

```bash
# Single case:
python optimize_cma.py --base_path ./data/different_types --case_name your_case_name --train_frame 43

# All cases:
python script_optimize.py
```
**Output:** `experiments_optimization/{case_name}/optimal_params.pkl`

### 4b. First-Order Optimization (Gradient-based)
~5 minutes per case

```bash
# Single case:
python train_warp.py --base_path ./data/different_types --case_name your_case_name --train_frame 43

# All cases:
python script_train.py
```
**Output:** `experiments/{case_name}/train/best_*.pth`

### 4c. Train Gaussian Appearance
```bash
bash gs_run.sh
```
**Output:** `gaussian_output/{case_name}/...`

---

## Step 5: Run Interactive Playground

```bash
python interactive_playground.py \
  --case_name your_case_name \
  --n_ctrl_parts 2  # 1 for single hand, 2 for two hands
  [--inv_ctrl]      # invert control direction (optional)
```

**Example commands:**
```bash
python interactive_playground.py --n_ctrl_parts 2 --case_name double_stretch_sloth
python interactive_playground.py --inv_ctrl --n_ctrl_parts 2 --case_name double_lift_cloth_3
```

---

## Configuration Files

Physics parameters are configured in YAML files:

| Config | Use Case | Key Parameters |
|--------|----------|----------------|
| `configs/real.yaml` | Stuffed toys, ropes | Higher spring stiffness, uses shape prior |
| `configs/cloth.yaml` | Flat cloth objects | Lower stiffness, self-collision enabled |

The config is **auto-selected** based on case name:
- Contains "cloth" or "package" → `cloth.yaml`
- Otherwise → `real.yaml`

### Key physics parameters in `real.yaml`:
```yaml
FPS: 30
dt: 5e-5
num_substeps: 667
init_spring_Y: 3e4        # Initial spring stiffness
collide_elas: 0.5         # Collision elasticity
collide_fric: 0.3         # Collision friction
object_radius: 0.02       # Particle radius for object
controller_radius: 0.04   # Particle radius for controller (hand)
```

---

## Visualization Tools

### Force Visualization
Visualize forces applied by hands to objects:
```bash
python visualize_force.py --case_name double_stretch_sloth --n_ctrl_parts 2
```
**Output:** `experiments/{case_name}/force_visualization.mp4`

### Material Visualization
Visualize material properties:
```bash
python visualize_material.py --case_name double_lift_cloth_1
```

### Render Evaluation
```bash
bash gs_run_simulate.sh           # Render dynamic videos
python export_render_eval_data.py
bash evaluate.sh                  # Get quantitative results
python visualize_render_results.py # Get qualitative results
```

---

## Directory Structure Summary

```
YOUR RAW DATA (minimum required)
├── color/
│   ├── 0/0.png, 1.png, ... (camera 0 frames)
│   ├── 1/0.png, 1.png, ... (camera 1 frames)
│   └── 2/0.png, 1.png, ... (camera 2 frames)
├── depth/
│   ├── 0/0.png, 1.png, ...
│   ├── 1/0.png, 1.png, ...
│   └── 2/0.png, 1.png, ...
├── calibrate.pkl (list of 4×4 c2w matrices)
└── metadata.json (intrinsics, fps, resolution)

    ↓ python process_data.py

PROCESSED DATA (generated)
├── mask/           (segmentation masks)
├── cotracker/      (2D tracking)
├── pcd/            (3D point clouds)
├── shape/          (3D shape prior if enabled)
├── final_data.pkl  (training data)
├── split.json      (train/test split)
└── gt_track_3d.pkl (ground truth 3D tracking)

    ↓ python optimize_cma.py + train_warp.py

PHYSICS MODEL
├── experiments_optimization/{case}/optimal_params.pkl
└── experiments/{case}/train/best_*.pth

    ↓ bash gs_run.sh

GAUSSIAN MODEL
└── gaussian_output/{case}/point_cloud/iteration_10000/point_cloud.ply

    ↓ python interactive_playground.py

🎮 INTERACTIVE PLAYGROUND
```

---

## Troubleshooting

### Open3D Visualization Errors (Headless/WSL)
If you see errors like:
```
[Open3D WARNING] GLFW Error: Wayland: Failed to load libwayland-cursor
[Open3D WARNING] Failed to create window
```

**Solution:** Set CPU rendering mode:
```bash
export OPEN3D_CPU_RENDERING=true
python visualize_force.py --case_name your_case_name
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `process_data.py` | Main data processing orchestrator |
| `script_process_data.py` | Batch process all cases |
| `optimize_cma.py` | Zero-order CMA-ES optimization |
| `train_warp.py` | First-order gradient optimization |
| `interactive_playground.py` | Real-time interactive simulation |
| `visualize_force.py` | Force visualization |
| `visualize_material.py` | Material property visualization |
| `gs_train.py` | Gaussian splatting training |
| `qqtt/engine/trainer_warp.py` | Core physics trainer using Warp |

---

## Detailed Script Explanations

### `script_optimize.py` (Stage 1: CMA-ES Optimization)

**Purpose:** Find good initial physics parameters using sampling-based optimization (no gradients needed)

**Algorithm:** CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- Samples random parameter combinations
- Simulates physics for each sample
- Compares simulated positions to tracked ground truth
- Updates search distribution based on lowest-error samples
- Repeats for `max_iter` iterations (default 20)

**Parameters optimized (12 total):**

| Parameter | Description | Range |
|-----------|-------------|-------|
| `global_spring_Y` | Spring stiffness | 0 - 100,000 |
| `object_radius` | Radius for connecting object particles | 0.01 - 0.05 |
| `object_max_neighbours` | Max spring connections per particle | 10 - 50 |
| `controller_radius` | Radius for hand-to-object connections | 0.01 - 0.08 |
| `controller_max_neighbours` | Max hand-object springs | 10 - 80 |
| `collide_elas` | Ground collision elasticity | 0 - 1 |
| `collide_fric` | Ground collision friction | 0 - 2 |
| `collide_object_elas` | Object self-collision elasticity | 0 - 1 |
| `collide_object_fric` | Object self-collision friction | 0 - 2 |
| `collision_dist` | Collision detection distance | 0.01 - 0.05 |
| `drag_damping` | Air drag | 0 - 20 |
| `dashpot_damping` | Spring damping | 0 - 200 |

**Time:** ~12 minutes per case

**Output:**
```
experiments_optimization/{case_name}/
├── optimal_params.pkl      # Best parameters found
├── optimizeCMA/
│   └── optimal.mp4         # Visualization of best simulation
└── optimize_cma_log.txt    # Log file
```

---

### `script_train.py` (Stage 2: Gradient-Based Optimization)

**Purpose:** Fine-tune physics parameters using gradient descent through differentiable simulation (NVIDIA Warp)

**How it works:**
1. Loads Stage 1 results from `optimal_params.pkl`
2. Initializes spring-mass system with those parameters
3. Runs differentiable simulation using NVIDIA Warp
4. Computes loss by comparing simulated vs tracked positions
5. Backpropagates gradients through the physics simulation
6. Updates parameters (now per-spring, not global)
7. Saves checkpoints during training

**What it optimizes:**
- Per-spring stiffness values (`spring_Y`) - individualized for each spring
- Collision elasticity/friction
- Other learnable physics parameters

**Time:** ~5 minutes per case

**Output:**
```
experiments/{case_name}/
├── train/
│   ├── best_199.pth        # Best model checkpoint
│   ├── iter_0.pth          # Intermediate checkpoints
│   ├── iter_100.pth
│   └── ...
├── init.mp4                # Initial simulation video
└── inv_phy_log.txt         # Training log
```

---

### `script_inference.py` (Evaluation)

**Purpose:** Run the trained model on test frames (not seen during training) and evaluate accuracy

**How it works:**
1. Loads Stage 1 params from `optimal_params.pkl`
2. Loads trained model from `experiments/{case_name}/train/best_*.pth`
3. Runs simulation on all frames (including test set)
4. Evaluates accuracy against ground truth tracking
5. Saves visualization videos

**Output:**
```
experiments/{case_name}/
├── inference/
│   ├── inference.mp4       # Simulation video
│   └── metrics.json        # Quantitative results (Chamfer distance, etc.)
└── inference_log.txt
```

---

### Physics Training Pipeline Summary

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  script_optimize.py  │ --> │  script_train.py     │ --> │  script_inference.py │
│  (CMA-ES ~12 min)    │     │  (Gradient ~5 min)   │     │  (Evaluation)        │
├──────────────────────┤     ├──────────────────────┤     ├──────────────────────┤
│ Input:               │     │ Input:               │     │ Input:               │
│ final_data.pkl       │     │ optimal_params.pkl   │     │ best_*.pth           │
├──────────────────────┤     ├──────────────────────┤     ├──────────────────────┤
│ Output:              │     │ Output:              │     │ Output:              │
│ optimal_params.pkl   │     │ best_*.pth           │     │ inference.mp4        │
│ (12 global params)   │     │ (per-spring params)  │     │ metrics.json         │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

---

## Additional Notes

*(Add your own notes here as you explore the repo)*

---

## Alternative Depth Estimation (Video-Depth-Anything)

If you don't have RealSense depth cameras, you can use **Video-Depth-Anything (VDA)** to estimate depth from RGB videos. Two utility scripts are provided:

### `infer_depth.py` - Depth Inference from RGB

Runs Video-Depth-Anything on all cameras for cases in `data_config.csv`.

**Usage:**
```bash
# Process all cases (reads from data_config.csv)
python infer_depth.py

# Process specific case
python infer_depth.py --case_name double_stretch_sloth

# Process specific camera only
python infer_depth.py --case_name double_stretch_sloth --camera 1

# Use inverse depth output (recommended for VDA)
python infer_depth.py --case_name double_stretch_sloth --invert
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--case_name` | Specific case to process (otherwise all from data_config.csv) | None |
| `--camera` | Specific camera ID (0, 1, or 2) | All cameras |
| `--invert` | Invert depth output (VDA outputs inverse/disparity depth) | False |
| `--vda_path` | Path to Video-Depth-Anything repo | `../Video-Depth-Anything` |

**Output:**
```
data/vda_depth/{case_name}/
├── 0/                     # Camera 0 depth
│   ├── 0.npy, 1.npy, ...  # Per-frame depth (same format as RealSense)
│   └── depth_vis.mp4      # Visualization video
├── 1/                     # Camera 1 depth
└── 2/                     # Camera 2 depth
```

**Note:** VDA outputs *inverse depth* (closer = higher values). Use `--invert` to flip to standard depth format where closer = lower values.

---

### `compare_depth.py` - Depth Quality Comparison

Compare VDA predicted depth against RealSense reference depth using **relative metrics** (since monocular depth has arbitrary scale).

**Usage:**
```bash
# Compare all cameras for a case
python compare_depth.py --case_name double_stretch_sloth

# Compare specific camera
python compare_depth.py --case_name double_stretch_sloth --camera 0

# Limit frames for faster testing
python compare_depth.py --case_name double_stretch_sloth --max_frames 50

# Custom paths
python compare_depth.py --ref_dir path/to/realsense --pred_dir path/to/vda --output_dir results/
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--case_name` | Case to compare | Required (unless using custom paths) |
| `--camera` | Specific camera ID | All cameras |
| `--max_frames` | Max frames to process (-1 for all) | -1 |
| `--ref_dir` | Custom reference depth directory | Auto from case |
| `--pred_dir` | Custom predicted depth directory | Auto from case |
| `--output_dir` | Output directory | `depth_comparison/` |

**Output:**
```
depth_comparison/{case_name}/
├── comparison_camera_0.mp4   # Side-by-side video with RGB, depths, error map
├── comparison_camera_1.mp4
├── comparison_camera_2.mp4
└── metrics_summary.txt       # Per-frame and average metrics
```

---

### Depth Comparison Metrics Explained

The comparison uses **relative metrics** because monocular depth models output depth with arbitrary scale and shift. These metrics test structural correctness, not absolute accuracy.

#### Relative Metrics (Scale-Independent)

| Metric | Range | Good Value | Description |
|--------|-------|------------|-------------|
| **Spearman Rank** | -1 to 1 | > 0.90 | Are points ordered correctly by depth? Ranks all pixels near→far and checks if rankings match. |
| **Gradient Corr** | -1 to 1 | > 0.85 | Are depth edges preserved? Computes Sobel gradients and correlates them. |
| **Ordinal Error** | 0% to 50% | < 10% | % of point pairs with wrong depth ordering. Intuitive: "10% means 10% of comparisons are wrong." |

#### Aligned Metrics (After Scale+Shift Correction)

| Metric | Units | Good Value | Description |
|--------|-------|------------|-------------|
| **SI-RMSE** | meters | < 0.10 | Standard RMSE after optimal scale+shift alignment. |
| **Scale** | unitless | ~1.0 | Multiplier to convert predicted to reference depth. |
| **Shift** | meters | ~0.0 | Offset to add after scaling. |

#### Why These Metrics?

- **Spearman**: Tests if the model understands "A is closer than B" - the core capability of monocular depth.
- **Gradient Corr**: Catches blurry edges or boundary misalignments that Spearman might miss.
- **Ordinal Error**: More intuitive than correlation - directly answers "how often is the depth ordering wrong?"
- **SI-RMSE**: Fair comparison after removing scale ambiguity - shows actual metric error.

#### Interpreting Results

| Spearman | Quality | Interpretation |
|----------|---------|----------------|
| > 0.95 | Excellent | Nearly perfect depth ordering |
| 0.85-0.95 | Good | Minor ordering errors, usable |
| 0.70-0.85 | Fair | Noticeable errors, may need filtering |
| < 0.70 | Poor | Significant structural errors |

**Auto-Inversion:** The script automatically detects if predicted depth is inverted (Spearman < -0.5) and corrects it before computing other metrics.

---

### Workflow: Using VDA Depth Instead of RealSense

1. **Prepare RGB videos** in `data/different_types/{case_name}/color/{0,1,2}/`

2. **Run depth inference:**
   ```bash
   python infer_depth.py --case_name your_case --invert
   ```

3. **Align scales across cameras:**
   ```bash
   python align_depth.py --case_name your_case
   ```

4. **(Optional) Compare with RealSense if available:**
   ```bash
   python compare_depth.py --case_name your_case
   ```

5. **Copy VDA depth to expected location:**
   ```bash
   # Replace RealSense depth with VDA depth
   cp -r data/vda_depth/your_case/* data/different_types/your_case/depth/
   ```

6. **Continue with normal pipeline:**
   ```bash
   python process_data.py --case_name your_case ...
   ```

**Note:** VDA depth may have lower accuracy than RealSense for close-range manipulation tasks. Check the comparison metrics before proceeding.

---

### `align_depth.py` - Multi-Camera Depth Scale Alignment

**Problem**: VDA runs independently on each camera, producing depth with arbitrary scale. When lifting to 3D, point clouds from different cameras won't be consistent - the same physical point will appear at different distances.

**Solution**: Find per-camera scale factors that maximize 3D point cloud alignment using cross-view consistency. **No baseline depth data required** - only needs camera calibration (which you have from the physical camera setup).

**Requirements:**
- VDA depth output (`data/vda_depth/{case}/`)
- Camera calibration (`metadata.json` for intrinsics, `calibrate.pkl` for extrinsics)
- **NO depth sensor data needed** - alignment is purely geometric

**Usage:**
```bash
# Align depth for a specific case
python align_depth.py --case_name double_stretch_sloth

# Align all cases from data_config.csv
python align_depth.py

# Use Chamfer distance method instead of cross-projection
python align_depth.py --case_name double_stretch_sloth --method chamfer

# Custom output directory
python align_depth.py --case_name double_stretch_sloth --output_dir data/vda_depth_aligned
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--case_name` | Specific case to process | All from data_config.csv |
| `--base_path` | Path to case data (for calibration) | `./data/different_types` |
| `--vda_path` | Path to VDA depth output | `./data/vda_depth` |
| `--output_dir` | Output directory for aligned depth | `vda_path + "_aligned"` |
| `--method` | Alignment method: `cross_projection` or `chamfer` | `cross_projection` |
| `--max_frames` | Max frames to use for optimization | -1 (all) |
| `--n_frames` | Number of frames to use for scale optimization (-1 for all) | -1 |

**Output:**
```
data/vda_depth_aligned/{case_name}/
├── 0/                  # Camera 0 (reference, scale=1.0)
│   └── *.npy           # Scaled depth files
├── 1/                  # Camera 1 (scale optimized)
│   └── *.npy
├── 2/                  # Camera 2 (scale optimized)
│   └── *.npy
└── scales.json         # Optimized scale factors
```

---

#### How Cross-Projection Alignment Works

The key insight: **if two cameras see the same 3D point, their depth estimates (after scaling) should agree when projected into each other's view.**

**Step-by-step for camera pair (i, j):**

```
Camera i                           Camera j
─────────                          ─────────
1. Sample pixel (u_i, v_i)         
   with depth d_i                  

2. Unproject to 3D world:
   X_cam = K_i^{-1} @ [u, v, 1]^T × (s_i × d_i)
   X_world = T_i @ X_cam
   
3. Project X_world into camera j:
   X_cam_j = T_j^{-1} @ X_world
   [u_j, v_j] = K_j @ X_cam_j / X_cam_j.z
   
4. Expected depth in cam j:        4. Actual depth in cam j:
   d_expected = X_cam_j.z             d_actual = depth_j[v_j, u_j] × s_j
   
5. Loss = (log(d_expected) - log(d_actual))²
```

**Why log-space?** Makes the optimization scale-invariant and creates a smoother landscape.

**Loss function:**
$$\mathcal{L}(s_1, s_2) = \frac{1}{|P|} \sum_{(i,j) \in \text{pairs}} \sum_{p \in P_{ij}} \left( \log(d_{\text{expected}}) - \log(d_{\text{actual}}) \right)^2$$

Where:
- $s_0 = 1.0$ (camera 0 is reference)
- $s_1, s_2$ are learnable scale factors
- $P_{ij}$ are sampled points visible in both cameras

**Optimization:**
- **Algorithm**: L-BFGS-B (quasi-Newton with bounds)
- **Bounds**: $s_i \in [0.1, 10.0]$
- **Samples**: 1000 random pixels per camera pair
- **Frames**: Uses 5 frames spread across the video

**Why this works without ground truth depth:**
- Camera intrinsics (K) and extrinsics (T) are known from calibration
- VDA depth has correct *relative* structure within each view
- The only unknown is the *scale* per camera
- Cross-projection creates equations that constrain these scales

---

#### Alternative: Chamfer Distance Method

Instead of cross-projection, directly minimize point cloud misalignment:

1. Unproject each camera's depth to 3D point cloud
2. Compute Chamfer distance between point cloud pairs
3. Optimize scales to minimize total Chamfer distance

**Chamfer distance:**
$$\text{Chamfer}(A, B) = \frac{1}{|A|}\sum_{a \in A} \min_{b \in B} \|a - b\| + \frac{1}{|B|}\sum_{b \in B} \min_{a \in A} \|a - b\|$$

**Pros/Cons:**
| Method | Pros | Cons |
|--------|------|------|
| Cross-projection | Faster, more direct | Requires overlapping views |
| Chamfer | More robust to occlusion | Slower, noisier gradients |

---

**Updated Workflow with Alignment:**

```bash
# 1. Infer depth from RGB (no depth sensor needed)
python infer_depth.py --case_name your_case --invert

# 2. Align scales across cameras (uses only calibration)
python align_depth.py --case_name your_case

# 3. (Optional) Compare with RealSense if available for verification
python process_data.py --case_name your_case ...
```
