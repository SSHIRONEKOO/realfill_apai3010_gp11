# RealFill: Personalized Text-to-Image Inpainting

**APAI3010 Course Project (Group 11)**

This repository is an unofficial RealFill implementation based on the paper [RealFill: Reference-Driven Generation for Authentic Image Completion](https://arxiv.org/abs/2309.16668) and the base project [thuanz123/realfill](https://github.com/thuanz123/realfill)

The main Colab walkthrough is in [3010_project.ipynb](./3010_project_cleared_colab.ipynb). It covers setup, training, inference, visualization, and export.

---

## What This Project Does

RealFill personalizes a Stable Diffusion inpainting model using only 1–5 reference images of a scene. The notebook fine-tunes LoRA weights on both the UNet and the text encoder using `diffusers` + `peft`.

The notebook is configured for Google Colab (GPU T4), but the same scripts also run locally if you have a CUDA GPU. Change the directory if you want to run locally.

---

## Repository Structure

```sh
├── 3010_project_cleared_colab.ipynb   # Main Colab notebook
├── train_realfill.py                  # Training script
├── infer.py                           # Inference script
├── data/                              # Scene folders: ref/ + target/
├── requirements.txt                   # Core dependencies
└── README.md
```

## Notebook Workflow

The notebook follows this sequence:

1. Clone the repo and install dependencies
2. Configure `accelerate`
3. Train RealFill with LoRA on `data/<scene>/`
4. Run inference on `target.png` + `mask.png`
5. Visualize the 16 generated outputs
6. Zip and download the results

## Setup

In Colab, the notebook installs the required packages directly. For a local setup, use the same core dependencies shown in the notebook:

```bash
pip install diffusers accelerate transformers peft huggingface-hub torch torchvision ftfy tensorboard Jinja2 bitsandbytes xformers kornia
pip install matplotlib
```

You also need a Hugging Face token with access to `stabilityai/stable-diffusion-2-inpainting`.

## Training

The notebook trains with these key settings (same as default):

- base model: `sd2-community/stable-diffusion-2-inpainting`
- image resolution: `512`
- batch size: `16`
- max steps: `2000`
- LoRA rank / dropout / alpha: `8 / 0.1 / 16`
- mixed precision: `fp16`
- optimizations: gradient checkpointing, 8-bit Adam, xFormers, `set_grads_to_none`

Example command:

```bash
accelerate launch train_realfill.py \
  --pretrained_model_name_or_path sd2-community/stable-diffusion-2-inpainting \
  --train_data_dir data/20 \
  --output_dir 20-model \
  --resolution 512 \
  --train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --set_grads_to_none \
  --mixed_precision fp16 \
  --unet_learning_rate 2e-4 \
  --text_encoder_learning_rate 4e-5 \
  --lr_scheduler constant \
  --lr_warmup_steps 100 \
  --max_train_steps 2000 \
  --lora_rank 8 \
  --lora_dropout 0.1 \
  --lora_alpha 16 \
  --resume_from_checkpoint latest \
  --allow_tf32 \
  --enable_xformers_memory_efficient_attention
```

## Inference

The notebook runs inference with `infer.py` and saves 16 outputs:

```bash
python infer.py \
  --model_path 20-model \
  --validation_image data/20/target/target.png \
  --validation_mask data/20/target/mask.png \
  --output_dir ./test-infer/ \
  --seed 40
```

`infer.py` loads the base inpainting pipeline, swaps in the trained UNet and text encoder weights, applies the mask, and writes `0.png` to `15.png` into `test-infer/`.

## Data Format

```text
data/<scene>/
├── ref/               # 1–5 reference photos
└── target/
    ├── target.png     # image with the missing region
    └── mask.png       # white = area to inpaint
```

## Output

After inference, the notebook:

- shows the 16 generated images in a 4×4 grid
- saves the outputs in `test-infer/`
- zips the folder for download

## Acknowledgements

- Original RealFill paper: Tang et al. (SIGGRAPH 2024)
- Base implementation: [thuanz123/realfill](https://github.com/thuanz123/realfill)
- Libraries: PyTorch, Diffusers, Transformers, PEFT, bitsandbytes, xFormers, Accelerate
