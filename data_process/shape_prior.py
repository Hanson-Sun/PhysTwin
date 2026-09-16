import os
import sys
from pathlib import Path

# Use the installed xFormers backend; FlashAttention is not required.
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import imageio
from PIL import Image

# Support both the repository layout used by this project
# (data_process/TRELLIS/trellis) and a separately installed TRELLIS package.
_trellis_checkout = Path(__file__).resolve().parent / "TRELLIS"
if (_trellis_checkout / "trellis").is_dir():
    sys.path.insert(0, str(_trellis_checkout))

try:
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import render_utils, postprocessing_utils
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "TRELLIS is not installed. Clone/setup the TRELLIS repository at "
        "data_process/TRELLIS (or install its Python package) before enabling "
        "shape-prior generation. Use --no_shape_prior to skip this optional stage."
    ) from error

import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

img_path = args.img_path
output_dir = args.output_dir

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

final_im = Image.open(img_path).convert("RGBA")
assert not np.all(np.array(final_im)[:, :, 3] == 255)

# Run the pipeline
outputs = pipeline.run(
    final_im,
)

video_gs = render_utils.render_video(outputs["gaussian"][0])["color"]
video_mesh = render_utils.render_video(outputs["mesh"][0])["normal"]
video = [
    np.concatenate([frame_gs, frame_mesh], axis=1)
    for frame_gs, frame_mesh in zip(video_gs, video_mesh)
]
imageio.mimsave(f"{output_dir}/visualization.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs["gaussian"][0],
    outputs["mesh"][0],
    # Optional parameters
    simplify=0.95,  # Ratio of triangles to remove in the simplification process
    texture_size=1024,  # Size of the texture used for the GLB
)
glb.export(f"{output_dir}/object.glb")

# Save Gaussians as PLY files
outputs["gaussian"][0].save_ply(f"{output_dir}/object.ply")
