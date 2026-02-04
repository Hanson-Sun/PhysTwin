from PIL import Image
import torch
from argparse import ArgumentParser
import cv2
import numpy as np
import gc
import os

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument("--mask_path", type=str, default=None)
parser.add_argument("--output_path", type=str)
parser.add_argument("--category", type=str)
parser.add_argument("--low_memory", action="store_true", default=False, help="Use CPU offloading to reduce GPU memory (~8GB)")
parser.add_argument("--ultra_low_memory", action="store_true", default=False, help="Extreme memory saving - runs mostly on CPU (~4GB GPU)")
parser.add_argument("--simple", action="store_true", default=False, help="Use simple bicubic upscaling instead of SD (no GPU needed)")
args = parser.parse_args()

img_path = args.img_path
mask_path = args.mask_path
output_path = args.output_path
category = args.category

# Set memory-friendly environment variables
if args.low_memory or args.ultra_low_memory:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# let's download an  image
low_res_img = Image.open(img_path).convert("RGB")
if mask_path is not None:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(mask.shape)
    bbox = np.argwhere(mask > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    print(bbox)
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    print(bbox)
    low_res_img = low_res_img.crop(bbox)  # type: ignore

if args.simple:
    # Simple 4x bicubic upscaling - no GPU needed
    w, h = low_res_img.size
    upscaled_image = low_res_img.resize((w * 4, h * 4), Image.BICUBIC)
    upscaled_image.save(output_path)
    print(f"Simple upscaling complete: {output_path}")
else:
    # Use Stable Diffusion upscaler
    from diffusers import StableDiffusionUpscalePipeline

    # Clear any existing GPU memory first
    gc.collect()
    torch.cuda.empty_cache()

    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"

    if args.ultra_low_memory:
        # Ultra low memory: use aggressive CPU offloading (slowest but lowest memory)
        print("Loading model in ultra-low memory mode (this will be slow)...")
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        pipeline.enable_model_cpu_offload()
        # pipeline.enable_sequential_cpu_offload()
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_attention_slicing("max")
        # VAE optimizations
        if hasattr(pipeline, 'vae'):
            pipeline.vae.enable_slicing()
            pipeline.vae.enable_tiling()
            print("VAE slicing and tiling enabled")

        num_inference_steps = 20  # Default is 50

    elif args.low_memory:
        # Low memory: keep model on GPU but use memory optimizations
        print("Loading model in low-memory mode (model stays on GPU)...")
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        pipeline = pipeline.to("cuda")
        # Use attention slicing to reduce memory during attention computation
        pipeline.enable_attention_slicing(1)
        # Use VAE slicing for large images
        #pipeline.enable_vae_slicing()
        # Enable tiled VAE decoding for lower memory usage
        #pipeline.enable_vae_tiling()
        num_inference_steps = 50  # Default

    else:
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipeline = pipeline.to("cuda")
        num_inference_steps = 50

    prompt = f"Hand manipulates a {category}."

    # Run inference
    upscaled_image = pipeline(
        prompt=prompt,
        image=low_res_img,
        num_inference_steps=num_inference_steps,
    ).images[0]

    upscaled_image.save(output_path)
    print(f"SD upscaling complete: {output_path}")

    # Clean up GPU memory aggressively
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
