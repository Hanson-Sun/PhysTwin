"""Preprocess depth clips: quantile-normalize DA3 and VDA to [0,1] space."""

import sys, traceback, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .utils import align_vda_to_da3


def load(path: str) -> torch.Tensor:
    p = Path(path)
    if p.suffix == '.npy': return torch.from_numpy(np.load(path)).float()
    if p.suffix == '.pt':  return torch.load(path).float()
    raise ValueError(f"Unsupported format: {p.suffix}")


def align_dimensions(da3, vda, rgb):
    """Trim/resize all inputs to the smallest common T, H, W."""
    T = min(da3.shape[0], vda.shape[0], rgb.shape[0])
    H = min(da3.shape[1], vda.shape[1], rgb.shape[1])
    W = min(da3.shape[2], vda.shape[2], rgb.shape[2])

    da3, vda, rgb = da3[:T], vda[:T], rgb[:T]

    def resize_depth(d, h, w):
        return F.interpolate(d.unsqueeze(1), (h, w),
                             mode='bilinear', align_corners=False).squeeze(1)

    if da3.shape[1:] != (H, W): da3 = resize_depth(da3, H, W)
    if vda.shape[1:] != (H, W): vda = resize_depth(vda, H, W)
    if rgb.shape[1:3] != (H, W):
        rgb = F.interpolate(rgb.permute(0, 3, 1, 2).float(),
                            (H, W), mode='bilinear',
                            align_corners=False).permute(0, 2, 3, 1)
    return da3, vda, rgb


def process_clip(clip_id, da3_path, vda_path, rgb_path, output_dir, low_pct=2.0, high_pct=98.0):
    print(f"\n── {clip_id} ──", flush=True)

    da3 = load(da3_path)
    vda = load(vda_path)
    rgb = load(rgb_path)
    print(f"  Loaded  DA3={da3.shape} range=[{da3.min():.3f}, {da3.max():.3f}]", flush=True)
    print(f"          VDA={vda.shape} range=[{vda.min():.3f}, {vda.max():.3f}]", flush=True)
    print(f"          RGB={rgb.shape}", flush=True)

    da3, vda, rgb = align_dimensions(da3, vda, rgb)
    print(f"  Aligned DA3={da3.shape} VDA={vda.shape} RGB={rgb.shape}", flush=True)


    vda_aligned, da3_normed = align_vda_to_da3(vda, da3,low_pct=low_pct,high_pct=high_pct,)
    print(f"  DA3 normed range:  [{da3_normed.min():.3f}, {da3_normed.max():.3f}]", flush=True)
    print(f"  VDA normed range:  [{vda_aligned.min():.3f}, {vda_aligned.max():.3f}]", flush=True)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{clip_id}_depth_raw.npy",         da3_normed.numpy())
    np.save(out / f"{clip_id}_depth_vda_aligned.npy", vda_aligned.numpy())
    np.save(out / f"{clip_id}_rgb.npy",               rgb.numpy())
    print(f"  Saved depth_raw (normed), depth_vda_aligned (normed), rgb",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clip-id')
    p.add_argument('--depth-da3')
    p.add_argument('--depth-vda')
    p.add_argument('--rgb')
    p.add_argument('--batch-dir')
    p.add_argument('--output-dir',   required=True)
    p.add_argument('--low-pct',      type=float, default=2.0,
                   help='Lower percentile for quantile normalization (default 2)')
    p.add_argument('--high-pct',     type=float, default=98.0,
                   help='Upper percentile for quantile normalization (default 98)')
    p.add_argument('--num-workers',  type=int, default=2)
    p.add_argument('--skip-existing', action='store_true')
    args = p.parse_args()

    if args.clip_id:
        clips = [{'clip_id': args.clip_id, 'da3': args.depth_da3,
                  'vda': args.depth_vda, 'rgb': args.rgb}]
    elif args.batch_dir:
        bp    = Path(args.batch_dir)
        clips = []
        for f in sorted(bp.glob('*_depth_da3.npy')):
            cid = f.stem.replace('_depth_da3', '')
            vda = bp / f"{cid}_depth_vda.npy"
            rgb = bp / f"{cid}_rgb.npy"
            if vda.exists() and rgb.exists():
                clips.append({'clip_id': cid, 'da3': str(f),
                               'vda': str(vda), 'rgb': str(rgb)})
        if not clips:
            print("No clips found."); return 1
    else:
        p.error('Specify --clip-id or --batch-dir')

    if args.skip_existing:
        out   = Path(args.output_dir)
        clips = [c for c in clips
                 if not (out / f"{c['clip_id']}_depth_raw.npy").exists()]
        print(f"Processing {len(clips)} clips (skip-existing enabled)")

    def run(c):
        try:
            process_clip(c['clip_id'], c['da3'], c['vda'], c['rgb'],
                         args.output_dir, args.low_pct, args.high_pct)
            return 1
        except Exception as e:
            print(f"  ❌ ERROR {c['clip_id']}:", flush=True)
            print(traceback.format_exc(), flush=True)
            return 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        results = list(ex.map(run, clips))

    ok = sum(results)
    print(f"\nDone: {ok}/{len(clips)} clips processed", flush=True)
    return 0 if ok == len(clips) else 1


if __name__ == '__main__':
    sys.exit(main())