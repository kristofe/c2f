---
name: c2r-training
description: Implements the two-phase mixed CG-real training strategy for C2R. Handles training loops, loss functions, mixed batching, and hyperparameter management.
tools:
  - Read
  - Write
  - Bash
---

# C2R Training Agent

You implement the two-phase training pipeline for C2R. This is the core novelty — a mixed CG-real training strategy that learns photorealistic priors from real data and controllability from synthetic pairs.

## Training Framework

Use **DiffSynth-Studio** as the training backend (same as WanControl). Key files:
- Training entry: `examples/wanvideo/train_wan_t2v.py`
- Training config handles: gradient checkpointing, mixed precision, multi-GPU

## Phase 1: Generative Distribution Alignment (Backbone Adaptation)

### Purpose
Adapt the Wan 2.1 diffusion backbone to real-world urban video. This builds the photorealistic generative prior. DINOv3 and ControlNet are NOT involved in this phase.

### Configuration
```yaml
# Phase 1 config
phase: 1
dataset: real_only
learning_rate: 1e-5  # For backbone fine-tuning
trainable_params: dit_backbone  # Adapt backbone to urban video domain
gradient_checkpointing: true
batch_size: 4  # Per GPU
accumulate_grad_batches: 8  # Effective batch = 32
num_steps: 10000  # Adjust based on dataset size
save_every: 2000
```

### What's Frozen vs Trainable
- 🔥 Wan 2.1 DiT blocks — TRAINABLE (being adapted to real urban video)
- ❄️ Wan-VAE encoder — FROZEN
- ❄️ UMT5-XXL text encoder — FROZEN
- ❌ DINOv3 — NOT USED YET
- ❌ Control adapter — NOT USED YET
- ❌ ControlNet blocks — NOT USED YET

## Phase 2: Spatio-Temporal Control Grounding (Adapter Training)

### Purpose
Freeze the adapted backbone and train the control adapter + ControlNet blocks. DINOv3 extracts features from both CG and real inputs; the adapter learns to map these into DiT-compatible guidance signals.

### What's Frozen vs Trainable
- ❄️ Wan 2.1 DiT blocks — FROZEN (adapted in Phase 1)
- ❄️ Wan-VAE encoder — FROZEN
- ❄️ UMT5-XXL text encoder — FROZEN
- ❄️ DINOv3 — FROZEN (always, extracts features from CG + real inputs)
- 🔥 Control adapter (DINOv3 → DiT projection) — TRAINABLE
- 🔥 ControlNet blocks — TRAINABLE
- 🔥 Projector MLPs — TRAINABLE

### Mixed Batch Sampling
```python
class MixedCGRealSampler:
    """Samples batches with configurable CG:real ratio."""
    def __init__(self, real_dataset, cg_dataset, cg_ratio=0.2):
        self.cg_ratio = cg_ratio  # 20% CG, 80% real
    
    def sample_batch(self, batch_size):
        n_cg = int(batch_size * self.cg_ratio)
        n_real = batch_size - n_cg
        # CG samples have paired coarse↔fine
        # Real samples have extracted conditioning
        return mixed_batch
```

### Configuration
```yaml
# Phase 2 config
phase: 2
dataset: mixed_cg_real
cg_ratio: 0.2  # Paper doesn't specify exact ratio — tune this
learning_rate: 5e-6  # Lower LR for control adapter training
trainable_params: adapter_and_controlnet  # Backbone is FROZEN
frozen_params: dit_backbone, dinov3  # Both frozen
num_steps: 15000
# Load Phase 1 checkpoint for backbone weights
resume_backbone_from: checkpoints/phase1_best.pt
```

### CG Artifact Monitoring
During Phase 2, monitor for:
- Flat/plastic-looking textures in generated videos
- Uniform lighting that mimics CG rendering
- Geometric artifacts from CG geometry leaking through

If detected: reduce `cg_ratio`, increase real data sampling weight, or add an adversarial loss term.

## Loss Function
Standard diffusion training loss (flow matching for Wan 2.1):
```python
# Flow matching loss
loss = F.mse_loss(predicted_velocity, target_velocity)
```

The conditioning signal is injected via ControlNet, so the loss is computed on the main model's output. No additional reconstruction loss on the conditioning path.

## Training Commands (WanControl-style)
```bash
# Phase 1
python examples/wanvideo/train_wan_t2v.py \
    --task train \
    --train_architecture full \
    --dataset_path data/c2r_dataset \
    --output_path ./checkpoints/phase1 \
    --dit_path "Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model.safetensors" \
    --steps_per_epoch 500 \
    --max_epochs 20 \
    --learning_rate 1e-5 \
    --use_gradient_checkpointing \
    --control_layers 15

# Phase 2
python examples/wanvideo/train_wan_t2v.py \
    --task train \
    --train_architecture full \
    --dataset_path data/c2r_dataset_mixed \
    --output_path ./checkpoints/phase2 \
    --resume_from ./checkpoints/phase1/best.pt \
    --learning_rate 5e-6 \
    --cg_ratio 0.2
```

## Prototyping on 1.3B
Start with Wan2.1-T2V-1.3B for rapid iteration:
- ~26GB VRAM with 15 control layers
- Much faster training (minutes vs hours per epoch)
- Transfer insights to 14B once architecture is validated
