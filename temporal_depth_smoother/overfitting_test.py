import torch
import torch.optim as optim
import numpy as np
import sys
import argparse
from pathlib import Path

sys.path.insert(0, '/root/digital_clone_v2')
from temporal_depth_smoother.model import TemporalDepthSmoother
from temporal_depth_smoother.losses import total_loss
from temporal_depth_smoother.config import get_default_config
from temporal_depth_smoother.inference import sliding_window_inference
import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Overfitting test for temporal depth smoother')
    parser.add_argument('--clip',   default=None,  help='Clip name/ID to use (partial match). Defaults to first clip.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--frames', type=int, default=16, help='Number of frames to load for training')
    return parser.parse_args()


def load_clip(data_dir: str, clip_name: str | None, device: torch.device, num_frames: int = 16) -> tuple:
    all_paths = sorted(glob.glob(f'{data_dir}/*_depth_raw.npy'))

    if clip_name is not None:
        matches = [p for p in all_paths if clip_name in p]
        if not matches:
            print(f"No clip matching '{clip_name}' found. Available clips:")
            for p in all_paths:
                print(f"  {p.split('/')[-1].replace('_depth_raw.npy', '')}")
            sys.exit(1)
        path = matches[0]
    else:
        path = all_paths[0]

    clip_id = path.split('/')[-1].replace('_depth_raw.npy', '')
    print(f"Using clip: {clip_id}\n")

    raw = torch.from_numpy(np.load(path)[:num_frames]).float().unsqueeze(0).to(device)
    vda = torch.from_numpy(np.load(f'{data_dir}/{clip_id}_depth_vda_aligned.npy')[:num_frames]).float().unsqueeze(0).to(device)
    rgb = torch.from_numpy(np.load(f'{data_dir}/{clip_id}_rgb.npy')[:num_frames]).float().unsqueeze(0).to(device)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    print(f"Loaded first {num_frames} frames")
    print(f"raw:  range=[{raw.min():.3f}, {raw.max():.3f}]")
    print(f"vda:  range=[{vda.min():.3f}, {vda.max():.3f}]")
    print(f"rgb:  range=[{rgb.min():.3f}, {rgb.max():.3f}]")

    return clip_id, raw, vda, rgb


def print_temporal_stats(raw: torch.Tensor, vda: torch.Tensor) -> tuple:
    raw_tgrad = (raw[:, 1:] - raw[:, :-1]).abs().mean().item()
    vda_tgrad = (vda[:, 1:] - vda[:, :-1]).abs().mean().item()
    print(f"\nTemporal gradient (frame-to-frame change):")
    print(f"  raw: {raw_tgrad:.5f}  (flicker level — model must suppress this)")
    print(f"  vda: {vda_tgrad:.5f}  (TGM target — model should match this)")
    print(f"  ratio: {raw_tgrad / (vda_tgrad + 1e-8):.2f}x  (how much flickerier DA3 is vs VDA)")
    return raw_tgrad, vda_tgrad


def run_overfit(model, optimizer, raw, vda, rgb, config, num_epochs, raw_tgrad):
    print(f"\n{'step':>6}  {'total':>8}  {'fidelity':>9}  {'ssim':>8}  {'tgm':>8}  {'tv':>8}  {'geometric':>10}  {'geo_grad':>10}  {'flicker':>8}  {'delta_mean':>10}  {'out_tgrad':>10}")
    print("-" * 108)

    for step in range(num_epochs):
        optimizer.zero_grad()
        depth_smooth = model(raw, rgb)
        losses = total_loss(
            depth_smooth, raw, vda, rgb,
            lambda_fidelity=config.training.lambda_fidelity,
            lambda_ssim=config.training.lambda_ssim,
            lambda_tgm=config.training.lambda_tgm,
            lambda_tv=config.training.lambda_tv,
            tv_k1_weight=config.training.tv_k1_weight,
            tv_k2_weight=config.training.tv_k2_weight,
            tgm_k1_weight=config.training.tgm_k1_weight,
            tgm_k2_weight=config.training.tgm_k2_weight,
            tgm_k3_weight=config.training.tgm_k3_weight,
            lambda_geometric=config.training.lambda_geometric,
            lambda_geometric_grad=config.training.lambda_geometric_grad,
        )
        losses['total'].backward()
        optimizer.step()

        if step % 20 == 0:
            with torch.no_grad():
                delta     = (depth_smooth - raw).abs().mean().item()
                out_tgrad = (depth_smooth[:, 1:] - depth_smooth[:, :-1]).abs().mean().item()
                flicker   = raw_tgrad - out_tgrad
            print(f"{step:>6}  {losses['total'].item():>8.5f}  "
                  f"{losses['fidelity'].item():>9.5f}  "
                  f"{losses['ssim'].item():>8.5f}  "
                  f"{losses['tgm'].item():>8.5f}  "
                  f"{losses['tv'].item():>8.5f}  "
                  f"{losses['geometric'].item():>10.5f}  "
                  f"{losses['geometric_grad'].item():>10.5f}  "
                  f"{flicker:>8.5f}  "
                  f"{delta:>10.6f}  "
                  f"{out_tgrad:>10.5f}")


def print_expected_behaviour(vda_tgrad: float):
    print(f"\n{'Expected behaviour':}")
    print(f"  fidelity  → near 0       (output stays close to DA3 values)")
    print(f"  tgm       → near 0       (output temporal gradient matches VDA's: {vda_tgrad:.5f})")
    print(f"  geometric → near 0       (spatial structure preserved)")
    print(f"  delta     → near 0       (output ≈ input — correct for a smoother)")
    print(f"  out_tgrad → near {vda_tgrad:.5f}  (should land between vda and raw temporal grad)")
    print(f"\nIf total loss does not drop to near 0 within ~100 steps → fundamental issue")


def run_inference(model, clip_id, raw, vda, rgb, raw_tgrad, vda_tgrad, output_dir: Path, window_size: int = 16, overlap: int = 4):
    print(f"\nRunning inference on clip '{clip_id}'...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        depth_smooth_final = sliding_window_inference(
            model,
            raw,
            rgb,
            window_size=window_size,
            overlap=overlap,
        )  # [1, T, H, W]

    depth_smooth_np = depth_smooth_final.squeeze(0).cpu().numpy()
    raw_np          = raw.squeeze(0).cpu().numpy()
    vda_np          = vda.squeeze(0).cpu().numpy()
    rgb_np          = rgb.squeeze(0).cpu().numpy()

    out_tgrad_final = (depth_smooth_final[:, 1:] - depth_smooth_final[:, :-1]).abs().mean().item()
    flicker_final   = raw_tgrad - out_tgrad_final

    print(f"  output tgrad : {out_tgrad_final:.5f}  (raw={raw_tgrad:.5f}, vda={vda_tgrad:.5f})")
    print(f"  flicker supp : {flicker_final:.5f}  ({'✓ suppressed' if flicker_final > 0 else '✗ added flicker'})")
    print(f"  delta (mean) : {(depth_smooth_final - raw).abs().mean().item():.6f}")

    np.save(output_dir / f'{clip_id}_depth_smooth.npy', depth_smooth_np)
    np.save(output_dir / f'{clip_id}_depth_raw.npy',    raw_np)
    np.save(output_dir / f'{clip_id}_depth_vda.npy',    vda_np)
    np.save(output_dir / f'{clip_id}_rgb.npy',          rgb_np)

    print(f"\nSaved to {output_dir}/")
    print(f"  {clip_id}_depth_smooth.npy  {depth_smooth_np.shape}")
    print(f"  {clip_id}_depth_raw.npy     {raw_np.shape}")
    print(f"  {clip_id}_depth_vda.npy     {vda_np.shape}")
    print(f"  {clip_id}_rgb.npy           {rgb_np.shape}")


if __name__ == '__main__':
    args   = parse_args()
    config = get_default_config()
    device = torch.device('cuda')

    data_dir   = '/root/digital_clone_v2/temporal_depth_output/preprocessed'
    output_dir = Path('/root/digital_clone_v2/temporal_depth_output/overfitting')

    clip_id, raw, vda, rgb = load_clip(data_dir, args.clip, device, num_frames=args.frames)
    raw_tgrad, vda_tgrad   = print_temporal_stats(raw, vda)

    model     = torch.compile(TemporalDepthSmoother(config=config.model).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    run_overfit(model, optimizer, raw, vda, rgb, config, args.epochs, raw_tgrad)
    print_expected_behaviour(vda_tgrad)
    run_inference(model, clip_id, raw, vda, rgb, raw_tgrad, vda_tgrad, output_dir, window_size=16, overlap=4)