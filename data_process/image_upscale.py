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
parser.add_argument("--quantize", action="store_true", default=False, help="Use 8-bit quantization - keeps model on GPU with ~50%% less memory (~6-8GB)")
parser.add_argument("--quantize4", action="store_true", default=False, help="Use 4-bit quantization - keeps model on GPU with ~75%% less memory (~4-5GB)")
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

    if args.quantize4 or args.quantize:
        # Quantization - keeps model on GPU with less memory
        try:
            from diffusers import PipelineQuantizationConfig

            if args.quantize4:
                print("Loading model with 4-bit quantization (~4-5GB VRAM)...")
                quantization_config = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}
                )
            else:
                print("Loading model with 8-bit quantization (~6-8GB VRAM)...")
                quantization_config = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_8bit",
                    quant_kwargs={"load_in_8bit": True}
                )

            pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
            )
            pipeline.enable_attention_slicing(1)
            #pipeline.enable_vae_tiling()
            num_inference_steps = 50

        except ImportError:
            # Older diffusers version - try transformers BitsAndBytesConfig
            print("PipelineQuantizationConfig not available, trying transformers quantization...")
            from transformers import BitsAndBytesConfig

            if args.quantize4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            else:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # Load just the text encoder with quantization, keep rest in fp16
            from transformers import CLIPTextModel
            text_encoder = CLIPTextModel.from_pretrained(
                model_id,
                subfolder="text_encoder",
                quantization_config=quantization_config,
                device_map="cuda",  # Keep on GPU
            )

            pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                model_id,
                text_encoder=text_encoder,
                torch_dtype=torch.float16,
            )
            pipeline = pipeline.to("cuda")  # Keep entire pipeline on GPU
            pipeline.enable_attention_slicing(1)
            #pipeline.enable_vae_tiling()
            num_inference_steps = 50

        except Exception as e:
            print(f"Quantization failed: {e}")
            print("Falling back to low_memory mode...")
            args.low_memory = True
            args.quantize = False
            args.quantize4 = False

    if not args.quantize and not args.quantize4:
        if args.ultra_low_memory:
            # Ultra low memory: use aggressive CPU offloading (slowest but lowest memory)
            print("Loading model in ultra-low memory mode (this will be slow)...")
            pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            # Model CPU offload - moves entire models to CPU, very slow
            # pipeline.enable_model_cpu_offload()
            # Maximum attention slicing
            pipeline.enable_attention_slicing(1)
            pipeline.enable_sequential_cpu_offload()
            # VAE optimizations
            #pipeline.enable_vae_slicing()
            #pipeline.enable_vae_tiling()
            # Reduce inference steps for less memory (trades quality for memory)
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
        num_inference_steps=num_inference_steps
    ).images[0]

    upscaled_image.save(output_path)
    print(f"SD upscaling complete: {output_path}")

    # Clean up GPU memory aggressively
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
