import torch
import torch.optim as optim
import numpy as np
import sys
sys.path.insert(0, '/root/digital_clone_v2')

from temporal_depth_smoother.model import TemporalDepthSmoother
from temporal_depth_smoother.losses import total_loss, compute_flicker_mask
from temporal_depth_smoother.config import get_default_config

config = get_default_config()
device = torch.device('cuda')

# Load just 8 frames from one clip
data_dir = '/root/digital_clone_v2/temporal_depth_output/preprocessed'
import glob
path = sorted(glob.glob(f'{data_dir}/*_depth_raw.npy'))[0]
clip_id = path.split('/')[-1].replace('_depth_raw.npy', '')
print(f"Using clip: {clip_id}\n")

raw = torch.from_numpy(np.load(path)[:8]).float().unsqueeze(0).to(device)
vda = torch.from_numpy(np.load(f'{data_dir}/{clip_id}_depth_vda_aligned.npy')[:8]).float().unsqueeze(0).to(device)
rgb = torch.from_numpy(np.load(f'{data_dir}/{clip_id}_rgb.npy')[:8]).float().unsqueeze(0).to(device)
if rgb.max() > 1.0:
    rgb = rgb / 255.0

# Show flicker mask coverage for this sample
mask = compute_flicker_mask(raw, vda, rgb)
print(f"Flicker mask coverage: {mask.mean().item()*100:.1f}%")
print(f"raw:  range=[{raw.min():.3f}, {raw.max():.3f}]")
print(f"vda:  range=[{vda.min():.3f}, {vda.max():.3f}]")
print(f"rgb:  range=[{rgb.min():.3f}, {rgb.max():.3f}]")

model = TemporalDepthSmoother(config=config.model).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"\n{'step':>6}  {'total':>8}  {'flicker':>8}  {'geometric':>10}  {'smooth':>8}  {'delta_mean':>10}")
print("-" * 65)

for step in range(200):
    optimizer.zero_grad()
    depth_smooth = model(raw, rgb)
    losses = total_loss(
        depth_smooth, raw, vda, rgb,
        lambda_flicker=1.0,
        lambda_geometric=3.0,
        lambda_smooth=0.1,
    )
    losses['total'].backward()
    optimizer.step()

    if step % 20 == 0:
        delta = (depth_smooth - raw).abs().mean().item()
        print(f"{step:>6}  {losses['total'].item():>8.5f}  "
              f"{losses['flicker'].item():>8.5f}  "
              f"{losses['geometric'].item():>10.5f}  "
              f"{losses['smooth'].item():>8.5f}  "
              f"{delta:>10.6f}")

print("\nExpected: total loss drops to near 0 within ~100 steps")
print("If not → model or loss has a fundamental issue")