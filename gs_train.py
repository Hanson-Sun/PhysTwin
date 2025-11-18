#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, depth_loss, normal_loss, anisotropic_loss
from gaussian_splatting.gaussian_renderer import render, network_gui
import sys
from gaussian_splatting.scene import Scene, GaussianModel
from gaussian_splatting.utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from gaussian_splatting.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # if iteration % 100 == 0:
        #     try:
        #         def _to_cpu_minmax(t):
        #             t = t.detach().cpu()
        #             return float(t.min()), float(t.max()), float(t.mean())

        #         cam_center = getattr(viewpoint_cam, "camera_center", None)
        #         if cam_center is None:
        #             cam_center = getattr(viewpoint_cam, "world_view_transform", None)
        #         K = getattr(viewpoint_cam, "intrinsics", None) or getattr(viewpoint_cam, "K", None)
        #         orig_img = getattr(viewpoint_cam, "original_image", None)
        #         print(f"[DBG] iteration={iteration} picked_cam_index={vind}")
        #         if cam_center is not None:
        #             try:
        #                 print(f"[DBG] cam_center (sample) = {cam_center[:3] if hasattr(cam_center,'shape') else cam_center}")
        #             except:
        #                 print(f"[DBG] cam_center = {cam_center}")
        #         if K is not None:
        #             try:
        #                 print(f"[DBG] intrinsics K = {K if isinstance(K, (list,tuple)) else (K.shape if hasattr(K,'shape') else K)}")
        #             except:
        #                 print(f"[DBG] intrinsics = {K}")
        #         if orig_img is None:
        #             print("[DBG] original_image is None")
        #         else:
        #             try:
        #                 mn, mx, me = _to_cpu_minmax(orig_img)
        #                 print(f"[DBG] original_image shape={tuple(orig_img.shape)} min={mn:.6f} max={mx:.6f} mean={me:.6f}")
        #             except Exception as e:
        #                 print(f"[DBG] original_image stats failed: {e}")
        #         try:
        #             # GaussianModel exposes xyz via property `get_xyz` (a Parameter)
        #             if hasattr(gaussians, "get_xyz"):
        #                 xyz_param = gaussians.get_xyz
        #                 # xyz_param is a Parameter; detach and move to CPU
        #                 if isinstance(xyz_param, torch.nn.Parameter) or hasattr(xyz_param, 'detach'):
        #                     xyz = xyz_param.detach()
        #                 else:
        #                     xyz = torch.tensor(xyz_param)
        #                 xyz_cpu = xyz.cpu()
        #                 print(f"[DBG] gaussians.xyz shape={tuple(xyz_cpu.shape)} min={float(xyz_cpu.min()):.6f} max={float(xyz_cpu.max()):.6f}")
        #                 sample_n = min(5, xyz_cpu.shape[0])
        #                 print(f"[DBG] gaussians.xyz sample={xyz_cpu[:sample_n].numpy()}")
        #                 try:
        #                     import numpy as np
        #                     xyz_full = xyz_cpu.numpy()
        #                     total_pts = xyz_full.shape[0]
        #                     # fraction of near-exact duplicates after rounding
        #                     rounded = np.round(xyz_full, 4)
        #                     uniques = np.unique(rounded, axis=0)
        #                     dup_frac = 1.0 - float(len(uniques)) / float(total_pts)
        #                     print(f"[DBG] dup_frac_rounded4={dup_frac:.6f} unique={len(uniques)} total={total_pts}")

        #                     # simple 2-means on a random sample to find if there are two clusters
        #                     sample_size = min(20000, total_pts)
        #                     rng = np.random.default_rng(42)
        #                     sidx = rng.choice(total_pts, sample_size, replace=False)
        #                     sample_pts = xyz_full[sidx]
        #                     # init centers
        #                     centers = np.vstack((sample_pts[0], sample_pts[sample_size//2]))
        #                     for _km in range(20):
        #                         d0 = np.linalg.norm(sample_pts - centers[0], axis=1)
        #                         d1 = np.linalg.norm(sample_pts - centers[1], axis=1)
        #                         mask = d0 < d1
        #                         if mask.sum() == 0 or (~mask).sum() == 0:
        #                             break
        #                         c0 = sample_pts[mask].mean(axis=0)
        #                         c1 = sample_pts[~mask].mean(axis=0)
        #                         if np.allclose(c0, centers[0]) and np.allclose(c1, centers[1]):
        #                             break
        #                         centers = np.vstack((c0, c1))
        #                     counts0 = int(mask.sum())
        #                     counts1 = int(sample_size - counts0)
        #                     sep = float(np.linalg.norm(centers[0] - centers[1]))
        #                     print(f"[DBG] kmeans2_sample counts={counts0},{counts1} centroids={[centers[0].tolist(), centers[1].tolist()]} separation={sep:.6f}")
        #                 except Exception as e:
        #                     print(f"[DBG] gaussians cluster failed: {e}")
        #             elif hasattr(gaussians, "means"):
        #                 xyz = gaussians.means
        #                 xyz_cpu = xyz.detach().cpu()
        #                 print(f"[DBG] gaussians.means shape={tuple(xyz_cpu.shape)} min={float(xyz_cpu.min()):.6f} max={float(xyz_cpu.max()):.6f}")
        #                 sample_n = min(5, xyz_cpu.shape[0])
        #                 print(f"[DBG] gaussians.means sample={xyz_cpu[:sample_n].numpy()}")
        #         except Exception as e:
        #             print(f"[DBG] gaussians dump failed: {e}")
        #         # quick camera-list duplication check
        #         try:
        #             cams = scene.getTrainCameras()
        #             print(f"[DBG] total_train_cameras={len(cams)}")
        #         except:
        #             pass
        #     except Exception as e:
        #         print(f"[DBG] camera debug failed: {e}")

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if dataset.disable_sh:
            override_color = gaussians.get_features_dc.squeeze()
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, override_color=override_color, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        
        image, depth, normal, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], \
            render_pkg["depth"], \
            render_pkg["normal"], \
            render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], \
            render_pkg["radii"]
        
        pred_seg = image[3:, ...]
        image = image[:3, ...]
        gt_image = viewpoint_cam.original_image.cuda()

        # Mask out occluded regions
        if viewpoint_cam.occ_mask is not None:

            occ_mask = viewpoint_cam.occ_mask.cuda()
            inv_occ_mask = 1.0 - occ_mask
            
            # Expand inv_occ_mask to match each tensor shape
            image *= inv_occ_mask.unsqueeze(0)        # Shape: [3, 480, 848]
            # gt_image *= inv_occ_mask.unsqueeze(0)     # Shape: [3, 480, 848]
            pred_seg *= inv_occ_mask.unsqueeze(0)     # Shape: [1, 480, 848]
            depth *= inv_occ_mask                    # Shape: [480, 848]
            if normal is not None:
                normal *= inv_occ_mask.unsqueeze(-1)      # Shape: [480, 848, 3]

        # Loss
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            # image *= alpha_mask
            gt_image *= alpha_mask
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Segmentation Loss
        loss_seg = torch.tensor(0.0, device="cuda")
        if opt.lambda_seg > 0 and viewpoint_cam.alpha_mask is not None:
            gt_seg = viewpoint_cam.alpha_mask.cuda()
            loss_seg_l1 = l1_loss(pred_seg, gt_seg)
            loss_seg_ssim = ssim(image, gt_image)
            loss_seg = (1.0 - opt.lambda_dssim) * loss_seg_l1 + opt.lambda_dssim * (1.0 - loss_seg_ssim)
            loss = loss + opt.lambda_seg * loss_seg

        # Depth Loss
        loss_depth = torch.tensor(0.0, device="cuda")
        if opt.lambda_depth > 0:
            gt_depth = viewpoint_cam.depth.cuda()
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                loss_depth = depth_loss(depth, gt_depth, alpha_mask)
            else:
                loss_depth = depth_loss(depth, gt_depth)
            loss = loss + opt.lambda_depth * loss_depth

        # Normal Loss (rendered normals & normals estimated from omnidata)
        loss_normal = torch.tensor(0.0, device="cuda")
        if opt.lambda_normal > 0:
            gt_normal = viewpoint_cam.normal.cuda()
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                loss_normal = normal_loss(normal, gt_normal, alpha_mask)
            else:
                loss_normal = normal_loss(normal, gt_normal)
            loss = loss + opt.lambda_normal * loss_normal

        # Anisotropic Loss
        loss_anisotropic = torch.tensor(0.0, device="cuda")
        if opt.lambda_anisotropic > 0:
            loss_anisotropic = anisotropic_loss(gaussians.get_scaling)
            loss = loss + opt.lambda_anisotropic * loss_anisotropic

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                # progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss (no used)": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "L1 Loss": f"{Ll1.item():.{5}f}",
                                          "Depth Loss": f"{loss_depth.item():.{5}f}", "Normal Loss": f"{loss_normal.item():.{5}f}", 
                                          "Seg Loss": f"{loss_seg.item():.{5}f}", "Anisotropic Loss": f"{loss_anisotropic.item():.{5}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.train_test_exp, SPARSE_ADAM_AVAILABLE), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1], use_gsplat=True)  # default using gsplat

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
