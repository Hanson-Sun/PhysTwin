#!/usr/bin/env bash
# Re-applies the local modifications stored in this directory to the submodule
# working trees. Run from the repository root:
#   bash submodule_patches/apply_patches.sh
set -euo pipefail
cd "$(dirname "$0")/.."

patches_dir="$(cd submodule_patches && pwd)"
for patch_file in submodule_patches/*.patch; do
  sub="${patch_file#submodule_patches/}"
  sub="${sub%.patch}"
  abs_patch="$patches_dir/$(basename "$patch_file")"
  if git -C "$sub" apply --check "$abs_patch" 2>/dev/null; then
    echo "==> Applying $patch_file to $sub/"
    git -C "$sub" apply "$abs_patch"
  elif git -C "$sub" apply --reverse --check "$abs_patch" 2>/dev/null; then
    echo "==> $patch_file already applied to $sub/, skipping"
  else
    echo "ERROR: $patch_file does not apply cleanly to $sub/ (upstream may have changed)" >&2
    exit 1
  fi
done

echo "All submodule patches applied. Review with: git submodule foreach 'git status --short'"
