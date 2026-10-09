# Submodule patches

The `depth-inference-phystwin` branch gitlinks several third-party repos
(`Depth-Anything-3`, `GroundingDINO`, `SpaTrackerV2`, `TRELLIS`,
`Video-Depth-Anything`, `dust3r`, `mip-splatting`, `nvdiffrast`, ...), but there
is no `.gitmodules` file and we do not control forks of those upstreams. Any
edits made inside a submodule working tree are invisible to this repository, so
the meaningful local edits are captured here as patch files instead.

| Patch | Change |
| --- | --- |
| `Depth-Anything-3.patch` | Drop `random_state=42` from pose-alignment in `src/depth_anything_3/api.py` |
| `GroundingDINO.patch` | `Tensor.type().is_cuda()` -> `Tensor.is_cuda()` for newer PyTorch (CUDA extension sources) |
| `SpaTrackerV2.patch` | Add missing `__init__.py` files so `models.SpaTrackV2` is importable |
| `TRELLIS.patch` | Add `--no-build-isolation` to all `pip install` calls in `setup.sh` |
| `Video-Depth-Anything.patch` | Add `--low_memory` flag to `run.py`; download vitb weights in `get_weights.sh`; add `__init__.py` files |

## Applying

After cloning / `git submodule update --init` (or after updating a submodule to
a new upstream commit), run from the repo root:

```bash
bash submodule_patches/apply_patches.sh
```

To refresh these patches after editing a submodule, re-run from the repo root:

```bash
for d in Depth-Anything-3 GroundingDINO SpaTrackerV2 TRELLIS Video-Depth-Anything; do
  git -C "$d" diff > "submodule_patches/$d.patch"
done
```
