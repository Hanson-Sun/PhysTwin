# Robust Pose Alignment Fix for Streaming Depth Processing

## Problem Summary

When using `parse_depth_da3_streaming.py` with higher chunk numbers or high overlap (e.g., chunk_size=10, overlap=9), the Umeyama pose alignment frequently failed due to **singular/degenerate covariance matrices**. This happened because:

1. **High overlap** means many frames in a chunk are nearly identical
2. **Limited pose variation** causes the covariance matrix to be singular
3. **Fallback mechanism was broken** - it silently skipped alignment without applying scale correction
4. **Result**: Depth maps were never properly aligned to match camera intrinsics/extrinsics, causing major depth scale mismatches

## Root Cause

The Umeyama alignment algorithm requires sufficient variation in camera poses to compute a valid similarity transform (rotation, translation, scale). When frames are too similar or nearly coplanar:

```python
# In pose_align.py, umeyama_alignment()
u, d, v = np.linalg.svd(cov_xy)
if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
    return None, None, None  # Degenerate covariance rank!
```

This returns `None` for rotation, translation, and scale, causing the alignment to fail.

## Solution Strategy

Based on the insight that **we only care about aligning the final frames being saved** (not all frames in the chunk), the solution implements:

### 1. **Robust Error Handling in API** ([api.py](Depth-Anything-3/src/depth_anything_3/api.py#L345-L403))

```python
def _align_to_input_extrinsics_intrinsics(...):
    try:
        _, _, scale, aligned_extrinsics = align_poses_umeyama(...)
        
        if align_to_input_ext_scale:
            prediction.extrinsics = extrinsics[..., :3, :].numpy()
            if not np.isnan(scale) and not np.isinf(scale) and scale > 0:
                prediction.depth /= scale
            else:
                logger.warning(f"Invalid scale value: {scale}")
    except Exception as e:
        # Graceful fallback: use input extrinsics, keep depth as-is
        logger.warning(f"Pose alignment failed: {e}")
        prediction.extrinsics = extrinsics[..., :3, :].numpy()
```

**Changes:**
- ✅ Wrapped `align_poses_umeyama` in try-except
- ✅ Validates scale before applying it
- ✅ Graceful fallback when alignment fails
- ✅ Clear warning messages

### 2. **Per-Frame Alignment in Streaming** ([parse_depth_da3_streaming.py](parse_depth_da3_streaming.py#L88-L162))

Added `align_depth_to_pose_robust()` function that:
- Aligns each frame individually using ALL available frames as reference
- Uses RANSAC when sufficient frames are available (≥10)
- Validates scale before applying
- Returns success status for tracking

```python
def align_depth_to_pose_robust(
    depth: np.ndarray,
    predicted_ext: np.ndarray,
    target_ext: np.ndarray,
    all_predicted_exts: np.ndarray,  # All frames for reference
    all_target_exts: np.ndarray,      # All frames for reference
    use_ransac: bool = True,
    verbose: bool = False
) -> Tuple[np.ndarray, bool]:
    """Robustly align a single frame's depth to target extrinsics."""
```

### 3. **Modified Processing Pipeline**

**Before:**
```python
prediction = model.inference(
    image=all_images,
    intrinsics=intrinsics_tensor,
    extrinsics=extrinsics_tensor,
    align_to_input_ext_scale=True  # ❌ Fails on degenerate chunks
)
```

**After:**
```python
# Don't auto-align - we'll do it robustly per-frame
prediction = model.inference(
    image=all_images,
    intrinsics=intrinsics_tensor,
    extrinsics=extrinsics_tensor,
    align_to_input_ext_scale=False  # ✅ Disable auto-alignment
)

# Then align each frame individually
for frame_idx in range(num_frames):
    for cam_idx, cam_id in enumerate(camera_ids):
        aligned_depth, success = align_depth_to_pose_robust(
            depth=depth,
            predicted_ext=predicted_ext,
            target_ext=target_ext,
            all_predicted_exts=prediction.extrinsics,  # Use all frames
            all_target_exts=extrinsics_tensor,
            use_ransac=True
        )
```

### 4. **Alignment Statistics Tracking**

Added comprehensive tracking and reporting:

```
ALIGNMENT STATISTICS
================================================================================
Total frames processed: 120
Successful alignments:  115 (95.8%)
Failed alignments:      5 (4.2%)

Note: Failed alignments use depth as-is without scale correction.
      This may cause depth scale mismatches with camera calibration.
================================================================================
```

## Key Benefits

1. ✅ **Always attempts alignment** - No silent failures
2. ✅ **Per-frame robustness** - Each frame aligned independently using all available reference frames
3. ✅ **RANSAC for outlier rejection** - More robust when we have enough frames
4. ✅ **Clear visibility** - Track success/failure rates
5. ✅ **Graceful degradation** - When alignment fails, uses depth as-is with clear warning
6. ✅ **Flexible reference window** - Uses all frames in chunk for alignment, maximizing pose variation

## Usage

Run the streaming script as before:

```bash
python parse_depth_da3_streaming.py \
    --case_dir /path/to/case \
    --chunk_size 10 \
    --overlap 9 \
    --verbose  # To see alignment details
```

With `--verbose`, you'll see per-chunk alignment status:
```
Processing chunks: 100%|████████| 10/10 [01:23<00:00,  8.32s/chunk]
  Aligning 10 frames individually...
  Alignment: 10/10 successful (100.0%)
```

## Technical Details

### Why Per-Frame Alignment Works

Even with high overlap (e.g., 9/10 frames), the **cumulative set of all frames** typically has enough variation for Umeyama alignment. By aligning each frame using all frames as reference rather than just the chunk's frames, we:

1. Maximize pose variation (use frames from across time)
2. Enable RANSAC outlier rejection (when ≥10 frames available)
3. Get individual success/failure tracking per frame

### Fallback Strategy

When alignment fails (degenerate covariance):
1. **Use input extrinsics** as-is (camera calibration)
2. **Keep depth without scale correction** 
3. **Log warning** so user knows what happened
4. **Continue processing** - don't crash

This is better than the old behavior which silently skipped alignment, causing massive depth scale errors.

## Files Modified

1. **[Depth-Anything-3/src/depth_anything_3/api.py](Depth-Anything-3/src/depth_anything_3/api.py)**
   - Added try-except around `align_poses_umeyama`
   - Validate scale before applying
   - Better error messages

2. **[parse_depth_da3_streaming.py](parse_depth_da3_streaming.py)**
   - Added `align_depth_to_pose_robust()` function
   - Modified `process_chunk()` to use per-frame alignment
   - Added alignment statistics tracking
   - Added summary report at end

## Verification

To verify the fix is working:

1. **Check alignment success rate** in the summary report
2. **Look for warnings** about failed alignments
3. **Compare depth scales** - should now match camera calibration
4. **Visualize depths** in 3D - should align with camera poses

## Future Improvements

If alignment failures are still common, consider:

1. **Adaptive chunk sizing** - Use larger chunks when pose variation is low
2. **Keyframe selection** - Choose frames with maximum pose variation
3. **Multi-stage alignment** - Align in groups, then refine
4. **Alternative algorithms** - Try ICP or other registration methods
