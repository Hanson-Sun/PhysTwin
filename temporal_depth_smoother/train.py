"""Training script for temporal depth smoothing."""

import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import wandb

from .config import get_default_config
from .data import create_dataloaders
from .losses import total_loss
from .model import TemporalDepthSmoother
from .utils import load_checkpoint, save_checkpoint

# ---------------------------------------------------------------------------
# Train / val epochs
# ---------------------------------------------------------------------------


def train_epoch(
    model, loader, optimizer, config, device, epoch, scaler=None, global_step=0, baseline_ema=None
):
    model.train()
    total, counts = 0.0, 0
    loss_dict = {
        "fidelity": 0.0,
        "ssim": 0.0,
        "geometric": 0.0,
        "geometric_grad": 0.0,
        "tgm": 0.0,
        "tv": 0.0,
    }

    pbar = tqdm(
        loader,
        desc=f"  Train",
        leave=False,
        unit="batch",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )

    for i, batch in enumerate(pbar):
        depth_raw = batch["depth_raw"].to(device)
        rgb = batch["rgb"].to(device)
        depth_vda_aligned = batch["depth_vda_aligned"].to(device)

        optimizer.zero_grad()
        
        if baseline_ema is not None:
            baseline_ema.update(depth_raw, depth_vda_aligned)

        if scaler is not None:
            with autocast("cuda", dtype=torch.bfloat16):
                depth_smooth = model(depth_raw, rgb)
                losses = total_loss(
                    depth_smooth,
                    depth_raw,
                    depth_vda_aligned,
                    rgb,
                    lambda_fidelity=config.training.lambda_fidelity,
                    lambda_geometric=config.training.lambda_geometric,
                    lambda_geometric_grad=config.training.lambda_geometric_grad,
                    lambda_ssim=config.training.lambda_ssim,
                    lambda_tgm=config.training.lambda_tgm,
                    lambda_tv=config.training.lambda_tv,
                    tgm_baseline=baseline_ema.tgm if baseline_ema is not None else None,
                    tv_baseline=baseline_ema.tv if baseline_ema is not None else None,
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            raise NotImplementedError

        total += losses["total"].item()
        for k in loss_dict:
            loss_dict[k] += losses[k].item()
        counts += 1
        global_step += 1

        pbar.set_postfix(
            loss=f"{total / counts:.4f}",
            fid=f"{loss_dict['fidelity'] / counts:.4f}",
            ssim=f"{loss_dict['ssim'] / counts:.4f}",
            geom=f"{loss_dict['geometric'] / counts:.4f}",
            ggrad=f"{loss_dict['geometric_grad'] / counts:.4f}",
            tgm=f"{loss_dict['tgm'] / counts:.4f}",
            tv=f"{loss_dict['tv'] / counts:.4f}",
        )

        # Per-batch logging
        if (i + 1) % config.training.log_interval == 0:
            wandb.log(
                {
                    "batch/loss": losses["total"].item(),
                    "batch/fidelity": losses["fidelity"].item(),
                    "batch/ssim": losses["ssim"].item(),
                    "batch/geometric": losses["geometric"].item(),
                    "batch/geometric_grad": losses["geometric_grad"].item(),
                    "batch/tgm": losses["tgm"].item(),
                    "batch/tv": losses["tv"].item(),
                },
                step=global_step,
            )

    n = max(counts, 1)
    avg = total / n
    for k in loss_dict:
        loss_dict[k] /= n

    return avg, loss_dict, global_step


def val_epoch(model, loader, config, device):
    model.eval()
    total, counts = 0.0, 0
    loss_dict = {
        "fidelity": 0.0,
        "ssim": 0.0,
        "geometric": 0.0,
        "geometric_grad": 0.0,
        "tgm": 0.0,
        "tv": 0.0,
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Val  ", leave=False, unit="batch"):
            depth_raw = batch["depth_raw"].to(device)
            rgb = batch["rgb"].to(device)
            depth_vda_aligned = batch["depth_vda_aligned"].to(device)

            depth_smooth = model(depth_raw, rgb)
            losses = total_loss(
                depth_smooth,
                depth_raw,
                depth_vda_aligned,
                rgb,
                lambda_fidelity=config.training.lambda_fidelity,
                lambda_ssim=config.training.lambda_ssim,
                lambda_geometric=config.training.lambda_geometric,
                lambda_geometric_grad=config.training.lambda_geometric_grad,
                lambda_tgm=config.training.lambda_tgm,
                lambda_tv=config.training.lambda_tv,
            )
            total += losses["total"].item()
            for k in loss_dict:
                loss_dict[k] += losses[k].item()
            counts += 1

    n = max(counts, 1)
    avg = total / n
    for k in loss_dict:
        loss_dict[k] /= n
    return avg, loss_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    torch.set_float32_matmul_precision("high")

    p = argparse.ArgumentParser(description="Train temporal depth smoothing network")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default=None)
    p.add_argument("--wandb-run-id", default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_default_config()

    print(f"Device: {device}", flush=True)
    print(f"Temporal window: {config.model.T}", flush=True)
    print(f"Temporal dilations: {config.model.temporal_dilations}", flush=True)
    print(f"Base channels: {config.model.base_channels}", flush=True)
    print(f"Batch size: {config.training.batch_size}", flush=True)
    print(f"Learning rate: {config.training.learning_rate}", flush=True)
    print(f"Num epochs: {config.training.num_epochs}", flush=True)
    print(
        f"AMP (float16): {'ENABLED' if config.training.use_amp else 'DISABLED'}",
        flush=True,
    )
    print(
        f"\nℹ️  Edit config.py to change hyperparameters (not command-line args)\n",
        flush=True,
    )

    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                "model": vars(config.model),
                "training": vars(config.training),
                "data": vars(config.data),
            },
            f,
            indent=2,
        )

    print("Loading data...", flush=True)
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=config.training.batch_size,
        temporal_window=config.model.T,
        target_height=config.data.target_height,
        target_width=config.data.target_width,
    )
    print(
        f"Train={len(train_loader.dataset)}  Val={len(val_loader.dataset)}", flush=True
    )
    print(
        f"Data resolution: {config.data.target_height}x{config.data.target_width}",
        flush=True,
    )

    model = TemporalDepthSmoother(config=config.model).to(device)
    model = torch.compile(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = GradScaler("cuda") if config.training.use_amp else None

    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}", flush=True)
        start_epoch = load_checkpoint(model, optimizer, args.resume)
        global_step = start_epoch * len(train_loader)  # approximate

    wandb.init(
        project="temporal-depth-smoother",
        config={**vars(config.model), **vars(config.training), **vars(config.data)},
        resume="allow",
        id=args.wandb_run_id,
    )
    best_val = float("inf")

    epoch_pbar = tqdm(
        range(start_epoch, config.training.num_epochs),
        desc="Epochs",
        unit="epoch",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
    )
    

    for epoch in epoch_pbar:
        train_loss, train_loss_dict, global_step = train_epoch(
            model, train_loader, optimizer, config, device, epoch, scaler, global_step,
        )
        val_loss, val_loss_dict = val_epoch(model, val_loader, config, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/fidelity": train_loss_dict["fidelity"],
                "train/ssim": train_loss_dict["ssim"],
                "train/geometric": train_loss_dict["geometric"],
                "train/geometric_grad": train_loss_dict["geometric_grad"],
                "train/tgm": train_loss_dict["tgm"],
                "train/tv": train_loss_dict["tv"],
                "val/loss": val_loss,
                "val/fidelity": val_loss_dict["fidelity"],
                "val/ssim": val_loss_dict["ssim"],
                "val/geometric": val_loss_dict["geometric"],
                "val/geometric_grad": val_loss_dict["geometric_grad"],
                "val/tgm": val_loss_dict["tgm"],
                "val/tv": val_loss_dict["tv"],
                "lr": current_lr,
            },
            step=global_step,
        )

        is_best = val_loss < best_val
        epoch_pbar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            lr=f"{current_lr:.1e}",
            best="✓" if is_best else "",
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
        )

        if is_best:
            best_val = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, output_dir / "best_model.pt"
            )
            tqdm.write(f"  ✓ New best val loss: {best_val:.6f}  (epoch {epoch + 1})")
            wandb.run.summary["best_val_loss"] = best_val
            wandb.run.summary["best_epoch"] = epoch + 1

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
