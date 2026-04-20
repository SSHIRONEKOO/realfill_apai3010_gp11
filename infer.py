import argparse
import os

import torch
from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
    DDPMScheduler
)
from transformers import CLIPTextModel


def parse_args(argv=None):
    return _parser.parse_args(argv)


_parser = argparse.ArgumentParser(description="Inference")
_parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
_parser.add_argument(
    "--validation_image",
    type=str,
    default=None,
    required=True,
    help="The directory of the validation image",
)
_parser.add_argument(
    "--validation_mask",
    type=str,
    default=None,
    required=True,
    help="The directory of the validation mask",
)
_parser.add_argument(
    "--output_dir",
    type=str,
    default="./test-infer/",
    help="The output directory where predictions are saved",
)
_parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible inference.")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    generator = None


    _local_cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--sd2-community--stable-diffusion-2-inpainting/"
        "snapshots/5f74973cbb64c8568780732c17f43eb269d63a0d"
    )
    _base_model = _local_cache if os.path.exists(_local_cache) else "stabilityai/stable-diffusion-2-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        _base_model,
        torch_dtype=torch.float32,
        revision=None,
        local_files_only=(True if os.path.exists(_local_cache) else False),
    )

    pipe.unet = UNet2DConditionModel.from_pretrained(
        args.model_path, subfolder="unet", revision=None,
    )
    pipe.text_encoder = CLIPTextModel.from_pretrained(
        args.model_path, subfolder="text_encoder", revision=None,
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

    image = Image.open(args.validation_image).convert("RGB")
    mask_image = Image.open(args.validation_mask).convert("L")


    w, h = image.size
    new_w, new_h = w - (w % 8), h - (h % 8)
    if new_w != w or new_h != h:
        image = image.crop((0, 0, new_w, new_h))
        mask_image = mask_image.crop((0, 0, new_w, new_h))

    erode_kernel = ImageFilter.MaxFilter(3)
    mask_image = mask_image.filter(erode_kernel)

    blur_kernel = ImageFilter.BoxBlur(1)
    mask_image = mask_image.filter(blur_kernel)

    for idx in range(16):
        result = pipe(
            prompt="a photo of sks", image=image, mask_image=mask_image,
            num_inference_steps=200, guidance_scale=1, generator=generator,
        ).images[0]

        if result.size != image.size:
            result = result.resize(image.size, Image.LANCZOS)

        result = Image.composite(result, image, mask_image)
        result.save(f"{args.output_dir}/{idx}.png")

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main(parse_args())
