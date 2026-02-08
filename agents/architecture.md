---
name: c2r-architecture
description: Implements C2R model architecture including frozen DINOv3 feature extraction, control adapter, ControlNet-Transformer blocks, and conditioning module. Use when building or modifying model code.
tools:
  - Read
  - Write
  - Bash
---

# C2R Architecture Agent

You implement the C2R model architecture. Your scope covers the **frozen DINOv3 feature extractor**, the **learned control adapter** (DINOv3 → DiT guidance), the **ControlNet-Transformer blocks**, and the full forward pass.

## CRITICAL: DINOv3 is the Shared Feature Space

The paper explicitly uses **frozen DINOv3** (Meta's self-supervised ViT, arXiv 2508.10104) as the conditioning feature extractor. From Figure 2: _"a control adapter is trained to convert DINOv3 spatio-temporal features into an additive latent guidance signal."_

- DINOv3 is **FROZEN** — never trained, no gradients
- Both CG renderings and real video frames go through the same frozen DINOv3
- DINOv3's self-supervised features are naturally domain-invariant (trained on 1.7B diverse images)
- The **control adapter** is the learned component that maps DINOv3 features → DiT latent space

## Starting Points — READ THESE FIRST

Before writing any code, study these existing implementations:

1. **facebookresearch/dinov3**: The frozen feature extractor
   - Download ViT-L or ViT-H+ distilled variant (balance quality vs memory)
   - Features: dense patch-level embeddings with strong spatial semantics
   - Use frozen inference only — `model.eval()`, `torch.no_grad()`

2. **WanControl** (`shalfun/WanControl`): Reference ControlNet-Transformer for Wan2.1
   - `examples/wanvideo/train_wan_t2v.py` — training entry point
   - The `--control_layers` parameter controls how many blocks get ControlNet
   - Uses DiffSynth-Studio as the training backend

3. **TheDenk/wan2.1-dilated-controlnet**: Alternative ControlNet with dilated blocks
   - `wan_controlnet.py` — ControlNet module definition
   - `wan_transformer.py` — Modified transformer with ControlNet injection
   - 8 blocks for 1.3B model (stride=3), 6 blocks for 14B model (stride=4)

4. **PixArt-alpha/PixArt-alpha**: Original ControlNet-Transformer reference
   - `asset/docs/pixart_controlnet.md` — Architecture documentation
   - ControlNet is applied to first N blocks, output added to corresponding main block

## Architecture Implementation

### Step 1: DINOv3 Feature Extraction (FROZEN)
```python
class DINOv3FeatureExtractor(nn.Module):
    """Frozen DINOv3 encoder for extracting domain-invariant spatio-temporal features.
    
    Input: video frames [B, T, C, H, W]
    Output: dense features [B, T, num_patches, D_dino]
    
    DINOv3 is NEVER trained. It provides the shared feature space
    that makes CG→real transfer possible.
    """
    def __init__(self, model_name="dinov3_vitl14"):
        super().__init__()
        self.dino = torch.hub.load('facebookresearch/dinov3', model_name)
        self.dino.eval()
        for param in self.dino.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, video_frames):
        # video_frames: [B, T, C, H, W]
        B, T = video_frames.shape[:2]
        # Reshape to process all frames as a batch
        frames = video_frames.flatten(0, 1)  # [B*T, C, H, W]
        features = self.dino.forward_features(frames)
        patch_features = features["x_norm_patchtokens"]  # [B*T, num_patches, D_dino]
        # Reshape back to video format
        _, N, D = patch_features.shape
        return patch_features.view(B, T, N, D)
```

### Step 2: Control Adapter (LEARNED)
```python
class ControlAdapter(nn.Module):
    """Learned projection from DINOv3 feature space to DiT conditioning space.
    
    This is the bridge between frozen DINOv3 features and the ControlNet blocks.
    Maps DINOv3 patch features to match DiT hidden dimension and token format.
    """
    def __init__(self, dino_dim, dit_dim, num_heads=8):
        super().__init__()
        # Project DINOv3 features to DiT dimension
        self.proj = nn.Sequential(
            nn.Linear(dino_dim, dit_dim),
            nn.GELU(),
            nn.Linear(dit_dim, dit_dim),
        )
        # Temporal attention to aggregate spatio-temporal features
        self.temporal_attn = nn.MultiheadAttention(dit_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dit_dim)
    
    def forward(self, dino_features):
        # dino_features: [B, T, num_patches, D_dino]
        B, T, N, D = dino_features.shape
        x = self.proj(dino_features)  # [B, T, N, D_dit]
        # Reshape for spatio-temporal processing
        x = x.view(B, T * N, -1)  # [B, T*N, D_dit]
        x = self.norm(x + self.temporal_attn(x, x, x)[0])
        return x  # [B, T*N, D_dit] — matches DiT token sequence
```

### Step 3: ControlNet Block (from PIXART-δ)
```python
class C2RControlNetBlock(nn.Module):
    """Clone of a Wan 2.1 DiT block used as ControlNet."""
    def __init__(self, main_block):
        super().__init__()
        # Clone architecture from main block
        self.block = copy.deepcopy(main_block)
        # MLP projector: maps ControlNet features to main feature space
        hidden_dim = main_block.hidden_dim  # Get from actual Wan block
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # CRITICAL: Zero-init the last layer so ControlNet contributes nothing at start
        nn.init.zeros_(self.projector[-1].weight)
        nn.init.zeros_(self.projector[-1].bias)
```

### Step 4: Full C2R Model
```python
class C2RModel(nn.Module):
    """Wan 2.1 + frozen DINOv3 + learned control adapter + ControlNet blocks."""
    def __init__(self, wan_model, num_control_blocks, dino_dim=1024, conditioning_scale=1.0):
        self.wan = wan_model  # Frozen in Phase 2, adapted in Phase 1
        self.dino = DINOv3FeatureExtractor()  # Always frozen
        self.adapter = ControlAdapter(dino_dim, wan_model.hidden_dim)  # Learned
        self.controlnet_blocks = nn.ModuleList([
            C2RControlNetBlock(self.wan.blocks[i])
            for i in range(num_control_blocks)
        ])
        self.conditioning_scale = conditioning_scale  # User-controllable at inference
    
    def forward(self, noisy_latent, timestep, text_emb, condition_frames):
        # Extract DINOv3 features from conditioning frames (CG or real)
        dino_features = self.dino(condition_frames)  # [B, T, N, D_dino]
        control_tokens = self.adapter(dino_features)  # [B, T*N, D_dit]
        
        # Run ControlNet blocks on conditioning tokens
        control_outputs = []
        h = control_tokens
        for ctrl_block in self.controlnet_blocks:
            h = ctrl_block.block(h, timestep, text_emb)
            control_outputs.append(ctrl_block.projector(h) * self.conditioning_scale)
        
        # Run main DiT blocks with additive ControlNet guidance
        x = noisy_latent
        for i, main_block in enumerate(self.wan.blocks):
            x = main_block(x, timestep, text_emb)
            if i < len(control_outputs):
                x = x + control_outputs[i]  # Additive fusion
        return x
```

## Two-Phase Training — What's Frozen When

| Component | Phase 1 (Generative Alignment) | Phase 2 (Control Grounding) |
|-----------|-------------------------------|----------------------------|
| DINOv3 | FROZEN (always) | FROZEN (always) |
| Wan VAE encoder | FROZEN | FROZEN |
| Wan DiT backbone | **TRAINABLE** (adapted to real video) | **FROZEN** |
| Control adapter | Not used yet | **TRAINABLE** |
| ControlNet blocks | Not used yet | **TRAINABLE** |

## Critical Rules
- DINOv3 is ALWAYS frozen — `torch.no_grad()`, `model.eval()`, `requires_grad=False`
- NEVER modify the Wan 2.1 base model weights in Phase 2
- In Phase 2, ONLY the control adapter + ControlNet blocks are trainable
- Projector zero-init is non-negotiable — ensures training stability
- DINOv3 features from CG and real inputs should be structurally similar — verify this early
- Test tensor shapes exhaustively before training (DINOv3 patch count depends on resolution)
