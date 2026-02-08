---
name: c2r-integration
description: Sets up the development environment including Wan 2.1, WanControl, DiffSynth-Studio, and all dependencies. Use for environment setup, dependency management, and Docker configuration.
tools:
  - Read
  - Write
  - Bash
---

# C2R Integration Agent

You manage the development environment and dependency chain for C2R.

## Required Repositories

```bash
# 1. Wan 2.1 (base model)
git clone https://github.com/Wan-Video/Wan2.1.git
# Key: model definitions, VAE, text encoder integration

# 2. DINOv3 (frozen feature extractor — THE shared feature space)
git clone https://github.com/facebookresearch/dinov3.git
# Key: frozen ViT providing domain-invariant features for CG and real inputs
# This is NOT trained — used as-is for feature extraction

# 3. WanControl (ControlNet training for Wan)
git clone https://github.com/shalfun/WanControl.git
# Key: ControlNet-Transformer implementation, DiffSynth training integration

# 4. DiffSynth-Studio (training framework)
git clone https://github.com/modelscope/DiffSynth-Studio.git
# Key: split training, VRAM optimization, LoRA/ControlNet training

# 5. TheDenk dilated ControlNet (alternative reference)
git clone https://github.com/TheDenk/wan2.1-dilated-controlnet.git
# Key: pre-trained ControlNet weights, inference pipeline

# 6. PixArt-alpha (ControlNet-Transformer reference)
git clone https://github.com/PixArt-alpha/PixArt-alpha.git
# Key: Original ControlNet-Transformer architecture documentation
```

## Model Weights

```bash
# Wan 2.1 T2V 1.3B (for prototyping)
# HuggingFace: Wan-AI/Wan2.1-T2V-1.3B-Diffusers

# Wan 2.1 T2V 14B (for production)
# HuggingFace: Wan-AI/Wan2.1-T2V-14B-Diffusers

# DINOv3 (frozen feature extractor — REQUIRED)
# Use ViT-L distilled variant for prototyping (smaller, faster)
# Use ViT-H+ for production quality
# HuggingFace: facebook/dinov3-vitl14 (or dinov3-vith-plus)
# Also loadable via: torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl14')

# Pre-trained ControlNet weights (for validation)
# HuggingFace: TheDenk/wan2.1-t2v-1.3b-controlnet-canny-v1
# HuggingFace: TheDenk/wan2.1-t2v-14b-controlnet-depth-v1
```

## Python Environment

```bash
conda create -n c2r python=3.10 -y
conda activate c2r

# Core
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate

# DiffSynth-Studio
cd DiffSynth-Studio && pip install -e .

# WanControl dependencies
pip install deepspeed lightning

# Video processing
pip install decord opencv-python imageio[ffmpeg]

# VLM for captioning (optional, can run separately)
pip install qwen-vl-utils
```

## Verification Steps

1. **Wan 2.1 forward pass:**
```python
from diffusers import AutoencoderKLWan, WanTransformer3DModel
vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae")
transformer = WanTransformer3DModel.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer")
print(f"Transformer blocks: {len(transformer.transformer_blocks)}")
# Expected: 30 blocks for 1.3B
```

2. **WanControl training smoke test:**
```bash
python examples/wanvideo/train_wan_t2v.py \
    --task train --train_architecture full \
    --dataset_path data/example_dataset \
    --dit_path "..." --steps_per_epoch 5 --max_epochs 1 \
    --control_layers 5 --learning_rate 1e-5
```

3. **TheDenk ControlNet inference test:**
```bash
cd wan2.1-dilated-controlnet
python -m inference.cli_demo \
    --video_path "resources/test.mp4" \
    --prompt "test" \
    --controlnet_type "canny" \
    --base_model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --controlnet_model_path TheDenk/wan2.1-t2v-1.3b-controlnet-canny-v1
```

## GPU Requirements

| Configuration | VRAM | Use case |
|--------------|------|----------|
| 1.3B + 15 control layers | ~26GB | Prototyping on single A100/H100 |
| 1.3B + 5 control layers | ~19GB | Prototyping on RTX 4090 |
| 14B + 6 control layers | ~80GB+ | Production training, multi-GPU |
| 14B inference | ~40GB | FP8 quantized inference |
