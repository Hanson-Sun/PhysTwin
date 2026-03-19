import torch
import torch.optim as optim
import numpy as np
import sys
sys.path.insert(0, '/root/digital_clone_v2')
from temporal_depth_smoother.model import TemporalDepthSmoother
from temporal_depth_smoother.losses import total_loss
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

print(f"raw:  range=[{raw.min():.3f}, {raw.max():.3f}]")
print(f"vda:  range=[{vda.min():.3f}, {vda.max():.3f}]")
print(f"rgb:  range=[{rgb.min():.3f}, {rgb.max():.3f}]")

raw_tgrad = (raw[:, 1:] - raw[:, :-1]).abs().mean().item()
vda_tgrad = (vda[:, 1:] - vda[:, :-1]).abs().mean().item()
print(f"\nTemporal gradient (frame-to-frame change):")
print(f"  raw: {raw_tgrad:.5f}  (flicker level — model must suppress this)")
print(f"  vda: {vda_tgrad:.5f}  (TGM target — model should match this)")
print(f"  ratio: {raw_tgrad / (vda_tgrad + 1e-8):.2f}x  (how much flickerier DA3 is vs VDA)")

model = TemporalDepthSmoother(config=config.model).to(device)
model = torch.compile(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"\n{'step':>6}  {'total':>8}  {'fidelity':>9}  {'tgm':>8}  {'tv':>8}  {'geometric':>10}  {'flicker':>8}  {'delta_mean':>10}  {'out_tgrad':>10}")
print("-" * 95)

for step in range(200):
    optimizer.zero_grad()
    depth_smooth = model(raw, rgb)
    losses = total_loss(
        depth_smooth, raw, vda, rgb,
        lambda_fidelity=config.training.lambda_fidelity,
        lambda_tgm=config.training.lambda_tgm,
        lambda_tv=config.training.lambda_tv,
        lambda_geometric=config.training.lambda_geometric,
    )
    losses['total'].backward()
    optimizer.step()

    if step % 20 == 0:
        with torch.no_grad():
            delta     = (depth_smooth - raw).abs().mean().item()
            out_tgrad = (depth_smooth[:, 1:] - depth_smooth[:, :-1]).abs().mean().item()
            # Flicker = raw temporal grad minus output temporal grad.
            # Positive means we're suppressing flicker; negative means we added it.
            flicker   = raw_tgrad - out_tgrad
        print(f"{step:>6}  {losses['total'].item():>8.5f}  "
              f"{losses['fidelity'].item():>9.5f}  "
              f"{losses['tgm'].item():>8.5f}  "
              f"{losses['tv'].item():>8.5f}  "
              f"{losses['geometric'].item():>10.5f}  "
              f"{flicker:>8.5f}  "
              f"{delta:>10.6f}  "
              f"{out_tgrad:>10.5f}")

print(f"\n{'Expected behaviour':}")
print(f"  fidelity  → near 0       (output stays close to DA3 values)")
print(f"  tgm       → near 0       (output temporal gradient matches VDA's: {vda_tgrad:.5f})")
print(f"  geometric → near 0       (spatial structure preserved)")
print(f"  delta     → near 0       (output ≈ input — correct for a smoother)")
print(f"  out_tgrad → near {vda_tgrad:.5f}  (should land between vda and raw temporal grad)")
print(f"\nIf total loss does not drop to near 0 within ~100 steps → fundamental issue")