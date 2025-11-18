# Hand Tracking Flow - Complete Walkthrough

## Overview
The hand tracking system uses **dense optical flow tracking** (CoTracker) combined with **semantic segmentation** to identify and track hand control points across video frames. These tracked points then drive the physics simulation.

---

## Stage 1: Data Capture
**Location:** Real-world video capture with multiple cameras
- **Input:** Multi-camera video footage of someone manipulating an object
- **Equipment:** 3 RGB cameras + depth sensors
- **Output:** Raw video frames from multiple viewpoints

---

## Stage 2: Semantic Segmentation
**Location:** `data_process/segment.py`, `data_process/segment_util_video.py`
- **Input:** Raw video frames
- **Process:**
  1. Detect the object being manipulated (e.g., sloth, cloth, rope)
  2. Segment the hand/body regions that are controlling the object
  3. Use `CONTROLLER_NAME = "hand"` to identify hand regions
  4. Generate binary masks for each frame indicating where hands are

- **Output:** 
  - Object mask: pixels belonging to the manipulated object
  - Controller mask: pixels belonging to the hand/body holding the object
  - Stored as: `{case_name}/mask/{camera_id}/{frame_id}.png`

---

## Stage 3: Dense Optical Flow Tracking
**Location:** `data_process/dense_track.py`
- **Input:** 
  - Video frames
  - Binary masks from Stage 2
  
- **Process:**
  1. Load video frame (from `color/{camera_id}.mp4`)
  2. Load the controller mask (hand region)
  3. Extract all pixel coordinates from the hand mask
  4. Use **CoTracker** (neural optical flow tracker) to track these pixels across frames:
     ```
     query_pixels = np.argwhere(mask)  # All pixels in hand region
     # Randomly sample 5000 query points from the hand mask
     # Track these 5000 points across all frames using CoTracker
     ```
  5. CoTracker returns the trajectory of each pixel across time
  6. For each frame, collect the tracked hand points

- **Output:**
  - Tracked 3D trajectories in pixel/image space
  - Stored as: `{case_name}/cotracker/` (per camera)

---

## Stage 4: 2D to 3D Projection
**Location:** `data_process/data_process_track.py`
- **Input:**
  - 2D pixel trajectories from CoTracker (per camera)
  - Camera calibration matrices (intrinsics/extrinsics)
  - Depth maps from depth sensors
  
- **Process:**
  1. For each tracked 2D point in each frame:
     ```python
     # Get the depth value at that pixel location
     depth = depth_map[y, x]
     
     # Project 2D pixel + depth to 3D world coordinates using camera matrix
     # 3D_point = camera_matrix^-1 × [u, v, depth]
     ```
  2. Separate the tracked points into two categories:
     - **Object points:** Points on the manipulated object
     - **Controller points:** Points on the hand/body (tracked markers)
  
  3. Filter for valid controller points:
     - Must be tracked consistently across most frames
     - Remove points with poor tracking quality
     - Keep only the hand markers that are reliable

- **Output:**
  - `controller_points`: Shape `(n_frames, n_control_points, 3)`
  - Example: (39 frames, 30 control points, 3D coordinates)

---

## Stage 5: Data Validation & Filtering
**Location:** `data_process/data_process_track.py` (lines 257-290)
- **Input:** Raw controller points from Stage 4
- **Process:**
  1. Check for NaN or invalid values
  2. Compute color for each point (rainbow color based on Y-coordinate)
  3. Calculate smoothness metrics
  4. Remove spurious/noisy tracking points
  
- **Output:**
  - Cleaned `controller_points`
  - Associated `controller_colors`
  - Stored in pickle: `track_process_data.pkl`

---

## Stage 6: Integration into Physics System
**Location:** `qqtt/data/real_data.py`
- **Input:** Processed tracking data from pickle file
- **Process:**
  ```python
  with open(data_path, "rb") as f:
      data = pickle.load(f)
  
  controller_points = data["controller_points"]  # (n_frames, n_control_points, 3)
  ```
- **Storage:** Tensors on GPU for simulation

---

## Stage 7: Spring-Mass System Initialization
**Location:** `qqtt/engine/cma_optimize_warp.py._init_start()` and `qqtt/model/diff_simulator/spring_mass_warp.py`

### 7a: Spring Connectivity
- **Input:** 
  - First frame controller points
  - Reconstructed object geometry (structure_points)
  
- **Process:**
  1. Build KD-tree of object surface points
  2. For each control point:
     ```python
     # Search for nearby object surface points within radius
     [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
         controller_point,
         controller_radius=0.07,  # 7cm search radius
         max_neighbours=50
     )
     
     # Create springs connecting control point to nearby surface points
     for j in idx:
         spring = (control_point_index, surface_point_index)
         rest_length = distance(control_point, surface_point)
         springs.append(spring)
     ```
  3. If no neighbors found within radius, fallback to nearest neighbor:
     ```python
     if len(idx) == 0:
         [k, idx, _] = pcd_tree.search_knn_vector_3d(controller_point, 1)
     ```

- **Output:**
  - Spring topology connecting all control points to object
  - Rest lengths (zero-strain spring lengths)
  - Mass distribution

### 7b: State Initialization
- Each control point gets a position, velocity, and forces vector
- All initialized to match the input tracking data

---

## Stage 8: Runtime Simulation
**Location:** `qqtt/model/diff_simulator/spring_mass_warp.py.step()`

### 8a: Control Point Update (each frame)
```python
def set_controller_target(frame_idx):
    # Interpolate hand position from previous to current frame
    for each substep in [0, num_substeps):
        t = step / num_substeps
        control_x[i] = (
            original_control_point[i] + 
            (target_control_point[i] - original_control_point[i]) * t
        )
```

### 8b: Force Computation
```python
# For each spring connected to a control point:
for each spring(control_i, object_j):
    displacement = object_x[j] - control_x[i]
    spring_force = spring_constant * (length - rest_length)
    force[j] += spring_force
```

### 8c: Physics Integration
- Apply forces to object points
- Update velocities and positions
- Handle collisions
- Apply constraints (ground, object-object collisions)

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: VIDEO CAPTURE                                          │
│ 3 RGB cameras + depth sensors + actor with tracking markers    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 2: SEMANTIC SEGMENTATION                                  │
│ Extract hand region masks using GroundedSAM + depth            │
│ Output: mask/{cam}/{frame}.png                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 3: DENSE TRACKING (CoTracker)                             │
│ Track 5000 pixels from hand mask across frames                 │
│ Output: Per-camera 2D trajectories                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 4: 2D→3D PROJECTION                                       │
│ Use camera calibration + depth to get 3D world coordinates    │
│ Separate: object_points vs controller_points                  │
│ Output: track_process_data.pkl with 3D trajectories           │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 5: VALIDATION & FILTERING                                 │
│ Remove noise, fill gaps, validate consistency                  │
│ Output: final_data.pkl with cleaned controller_points          │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 6: PHYSICS SIMULATION INITIALIZATION                      │
│ Load controller_points from pickle                             │
│ Build spring topology connecting hands to object               │
│ Initialize masses, velocities, forces                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 7: SIMULATION (Training/Inference)                        │
│ For each frame:                                                │
│   - Set controller target from tracked hand position           │
│   - Solve spring-mass dynamics                                 │
│   - Compute forces between springs and object                 │
│   - Update positions with physics integration                  │
│   - Apply constraints and collisions                           │
│ Output: Simulated cloth/object deformation                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 8: RENDERING                                              │
│ Use simulated mesh + Gaussian Splatting                        │
│ Output: Rendered videos matching real footage                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Data Structures

### Controller Points
```python
controller_points.shape = (n_frames, n_control_points, 3)
# Example: (39 frames, 30 control points, 3D coordinates)
# Each frame contains 30 tracked hand positions
```

### Springs
```python
springs[i] = [control_point_idx, object_point_idx]
# Spring connecting control point i to object surface point j
rest_lengths[i] = distance_at_rest
# Determines spring elasticity behavior
```

### Physics State (per frame)
```python
position_x[point_id] = (x, y, z)        # Current position
velocity_v[point_id] = (vx, vy, vz)     # Current velocity
forces_f[point_id] = (fx, fy, fz)       # Net force
mass_m[point_id] = scalar               # Point mass
```

---

## Why Your Control Points Appear "Clustered"

**They're actually well-distributed:**
- Two hand clusters (upper/lower) = expected for two-hand manipulation
- ~43cm vertical separation between hands
- ~20cm pairwise average distance
- Consistent smooth motion (5.4mm per frame on average)

**Visual clustering is just perspective:**
- Camera viewpoint compresses 3D data to 2D
- Hands' X and Z coordinates are similar
- Only Y coordinate differs significantly
- Appears as single blob on screen, but spatially separated

---

## Control Flow in Code

1. **Data Loading:** `qqtt/data/real_data.py`
2. **Initialization:** `qqtt/engine/trainer_warp.py.__init__()` or `qqtt/engine/cma_optimize_warp.py.__init__()`
3. **Spring Building:** `._init_start()` method
4. **Simulation:** `SpringMassSystemWarp.step()`
5. **Controller Update:** `set_controller_target(frame_idx)`
6. **Force Computation:** `eval_springs()` kernel in Warp
7. **Integration:** `integrate_ground_collision()` kernel

---

## Summary

**Hand tracking = Optical flow + 3D reconstruction + Physics constraints**

Your system:
1. ✅ Captures hand motion via video tracking
2. ✅ Converts 2D pixels to 3D world coordinates  
3. ✅ Identifies contact points between hands and object
4. ✅ Simulates object deformation driven by hand motion
5. ✅ Renders results back to video

The "clustering" appearance is normal for two-hand object manipulation!
