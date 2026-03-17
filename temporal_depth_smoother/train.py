"""Training script for temporal depth smoothing."""

import torch
import torch.optim as optim
import wandb
from torch.amp import autocast, GradScaler
import argparse
from pathlib import Path
import json
from tqdm import tqdm

from .config import get_default_config
from .model import TemporalDepthSmoother
from .data import create_dataloaders
from .losses import total_loss
from .utils import save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# Train / val epochs
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, config, device, epoch, scaler=None):
    model.train()
    total, counts = 0.0, 0
    loss_dict = {'temporal': 0.0, 'geometric': 0.0, 'smooth': 0.0}

    pbar = tqdm(loader, desc=f"  Train", leave=False, unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}")

    for i, batch in enumerate(pbar):
        depth_raw         = batch['depth_raw'].to(device)
        rgb               = batch['rgb'].to(device)
        depth_vda_aligned = batch['depth_vda_aligned'].to(device)

        optimizer.zero_grad()

        # Use automatic mixed precision if enabled
        if scaler is not None:
            with autocast('cuda', dtype=torch.float16):
                depth_smooth = model(depth_raw, rgb)
                losses = total_loss(
                    depth_smooth, depth_raw, depth_vda_aligned, rgb,
                    lambda_temporal=config.training.lambda_temporal,
                    lambda_geometric=config.training.lambda_geometric,
                    lambda_smooth=config.training.lambda_smooth,
                    motion_threshold=config.training.motion_threshold,
                )
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            depth_smooth = model(depth_raw, rgb)
            losses = total_loss(
                depth_smooth, depth_raw, depth_vda_aligned, rgb,
                lambda_temporal=config.training.lambda_temporal,
                lambda_geometric=config.training.lambda_geometric,
                lambda_smooth=config.training.lambda_smooth,
                motion_threshold=config.training.motion_threshold,
            )
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total += losses['total'].item()
        for k in loss_dict: loss_dict[k] += losses[k].item()
        counts += 1

        # Update tqdm postfix with running averages
        pbar.set_postfix(
            loss=f"{total/counts:.4f}",
            temp=f"{loss_dict['temporal']/counts:.4f}",
            geom=f"{loss_dict['geometric']/counts:.4f}",
            smooth=f"{loss_dict['smooth']/counts:.4f}",
        )

        # Per-batch logging every log_interval steps
        global_step = epoch * len(loader) + i
        if (i + 1) % config.training.log_interval == 0:
            wandb.log({
                'batch/loss':      losses['total'].item(),
                'batch/temporal':  losses['temporal'].item(),
                'batch/geometric': losses['geometric'].item(),
                'batch/smooth':    losses['smooth'].item(),
            }, step=global_step)

    n = max(counts, 1)
    avg = total / n
    for k in loss_dict: loss_dict[k] /= n

    wandb.log({
        'train/loss':      avg,
        'train/temporal':  loss_dict['temporal'],
        'train/geometric': loss_dict['geometric'],
        'train/smooth':    loss_dict['smooth'],
    }, step=epoch)

    return avg, loss_dict


def val_epoch(model, loader, config, device, epoch):
    model.eval()
    total, counts = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Val  ", leave=False, unit="batch"):
            depth_raw         = batch['depth_raw'].to(device)
            rgb               = batch['rgb'].to(device)
            depth_vda_aligned = batch['depth_vda_aligned'].to(device)

            depth_smooth = model(depth_raw, rgb)
            losses = total_loss(
                depth_smooth, depth_raw, depth_vda_aligned, rgb,
                lambda_temporal=config.training.lambda_temporal,
                lambda_geometric=config.training.lambda_geometric,
                lambda_smooth=config.training.lambda_smooth,
                motion_threshold=config.training.motion_threshold,
            )
            total += losses['total'].item()
            counts += 1

    avg = total / max(counts, 1)
    wandb.log({'val/loss': avg}, step=epoch)
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Train temporal depth smoothing network')
    p.add_argument('--data-dir',           required=True,
                   help='Path to preprocessed data directory')
    p.add_argument('--output-dir',         default='./outputs',
                   help='Directory to save checkpoints and logs')
    p.add_argument('--device',             default='cuda',
                   help='Device to use (cuda or cpu)')
    p.add_argument('--resume',             default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--wandb-run-id',       default=None,
                   help='W&B run ID to resume (leave blank to start a new run)')
    args = p.parse_args()

    device     = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config from config.py (source of truth for all hyperparameters)
    config = get_default_config()

    print(f"Device: {device}", flush=True)
    print(f"Temporal window: {config.model.T}", flush=True)
    print(f"Temporal dilations: {config.model.temporal_dilations}", flush=True)
    print(f"Base channels: {config.model.base_channels}", flush=True)
    print(f"Batch size: {config.training.batch_size}", flush=True)
    print(f"Learning rate: {config.training.learning_rate}", flush=True)
    print(f"Num epochs: {config.training.num_epochs}", flush=True)
    print(f"AMP (float16): {'ENABLED' if config.training.use_amp else 'DISABLED'}", flush=True)
    print(f"\nℹ️  Edit config.py to change hyperparameters (not command-line args)\n", flush=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump({'model': vars(config.model), 'training': vars(config.training),
                   'data': vars(config.data)}, f, indent=2)

    print("Loading data...", flush=True)
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=config.training.batch_size,
        temporal_window=config.model.T,
        target_height=config.data.target_height,
        target_width=config.data.target_width,
    )
    print(f"Train={len(train_loader.dataset)}  Val={len(val_loader.dataset)}", flush=True)
    print(f"Data resolution: {config.data.target_height}x{config.data.target_width}", flush=True)

    model = TemporalDepthSmoother(config=config.model).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate,
                           weight_decay=config.training.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    scaler = GradScaler('cuda') if config.training.use_amp else None

    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}", flush=True)
        start_epoch = load_checkpoint(model, optimizer, args.resume)

    wandb.init(
        project='temporal-depth-smoother',
        config={**vars(config.model), **vars(config.training), **vars(config.data)},
        resume='allow',
        id=args.wandb_run_id,
    )
    best_val = float('inf')

    # Outer epoch progress bar
    epoch_pbar = tqdm(range(start_epoch, config.training.num_epochs),
                      desc="Epochs", unit="epoch",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}")

    for epoch in epoch_pbar:
        train_loss, loss_dict = train_epoch(model, train_loader, optimizer, config, device, epoch, scaler)
        val_loss              = val_epoch(  model, val_loader,             config, device, epoch)

        scheduler.step(val_loss)
        wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)

        # Summarise epoch in the outer bar's postfix
        is_best = val_loss < best_val
        epoch_pbar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            best="✓" if is_best else "",
        )

        save_checkpoint(model, optimizer, epoch, val_loss,
                        output_dir / f"checkpoint_epoch_{epoch:03d}.pt")

        if is_best:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, output_dir / "best_model.pt")
            tqdm.write(f"  ✓ New best val loss: {best_val:.6f}  (epoch {epoch+1})")
            wandb.run.summary['best_val_loss'] = best_val
            wandb.run.summary['best_epoch']    = epoch + 1

    wandb.finish()
    print("\nDone!")


if __name__ == '__main__':
    main()