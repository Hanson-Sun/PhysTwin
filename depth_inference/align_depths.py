def reproject_depths_with_sim3(
    depths: Dict[int, np.ndarray],
    intrinsics: Dict[int, np.ndarray],
    extrinsics_inferred: Dict[int, np.ndarray],
    extrinsics_gt: Dict[int, np.ndarray],
    r: np.ndarray,
    t: np.ndarray,
    scale: float
) -> Dict[int, np.ndarray]:
    """
    Reproject depth maps after applying Sim(3) transformation.
    
    1. Backproject depth using inferred extrinsics → 3D world points
    2. Apply Sim(3) transformation (rotation, translation, scale)
    3. Reproject using ground truth extrinsics → corrected depths
    
    Args:
        depths: Dict camera_id → depth map (H, W)
        intrinsics: Dict camera_id → K (3, 3)
        extrinsics_inferred: Dict camera_id → w2c_inferred (4, 4)
        extrinsics_gt: Dict camera_id → w2c_gt (4, 4)
        r: Rotation matrix (3, 3) from Umeyama
        t: Translation vector (3,) from Umeyama
        scale: Scale factor from Umeyama
    
    Returns:
        Dict camera_id → corrected depth maps
    """
    reprojected = {}
    
    for cam_id, depth in depths.items():
        H, W = depth.shape
        depth = depth.astype(np.float32)
        
        K = intrinsics[cam_id]
        w2c_inferred = extrinsics_inferred[cam_id]
        w2c_gt = extrinsics_gt[cam_id]
        c2w_inferred = np.linalg.inv(w2c_inferred)
        
        # Step 1: Backproject to 3D world using inferred extrinsics
        K_inv = np.linalg.inv(K)
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        ones = np.ones((H, W))
        uv1 = np.stack([u, v, ones], axis=2)  # (H, W, 3)
        
        ray = np.einsum('ij,hwj->hwi', K_inv, uv1)  # (H, W, 3)
        X_c = ray[..., 0] * depth
        Y_c = ray[..., 1] * depth
        Z_c = ray[..., 2] * depth
        
        P_cam = np.stack([X_c, Y_c, Z_c, np.ones((H, W))], axis=2)  # (H, W, 4)
        P_world_inferred_homo = np.einsum('ij,hwj->hwi', c2w_inferred, P_cam)  # (H, W, 4)
        P_world_inferred = P_world_inferred_homo[..., :3]  # (H, W, 3)
        
        # Step 2: Apply Sim(3) transformation
        # P_corrected = scale * (r @ P_world_inferred) + t
        P_scaled = scale * P_world_inferred
        P_rotated = np.einsum('ij,hwj->hwi', r, P_scaled)  # (H, W, 3)
        P_world_gt = P_rotated + t[None, None, :]  # (H, W, 3)
        
        # Step 3: Reproject using ground truth extrinsics
        P_world_gt_homo = np.concatenate([P_world_gt, np.ones((H, W, 1))], axis=2)  # (H, W, 4)
        P_cam_gt_homo = np.einsum('ij,hwj->hwi', w2c_gt, P_world_gt_homo)  # (H, W, 4)
        P_cam_gt = P_cam_gt_homo[..., :3]  # (H, W, 3)
        
        # Extract depth (Z coordinate in camera frame)
        depth_corrected = P_cam_gt[..., 2]
        
        # Only keep valid depths (positive Z, matches original valid regions)
        valid = np.isfinite(depth) & (depth > 0)
        depth_corrected[~valid] = 0
        
        reprojected[cam_id] = depth_corrected.astype(depth.dtype)
    
    return reprojected