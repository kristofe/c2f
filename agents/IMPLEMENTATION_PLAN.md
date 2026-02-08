# C2R (Coarse-to-Real) — Implementation Plan & Multi-Agent System

## Paper Summary

**Paper:** "Coarse-to-Real: Generative Rendering for Populated Dynamic Scenes" (arXiv 2601.22301)
**Authors:** Gonzalo Gomez-Nogales¹, Yicong Hong², Chongjian Ge², Marc Comino-Trinidad¹, Dan Casas¹, Yi Zhou³
- ¹ Universidad Rey Juan Carlos
- ² Adobe Research
- ³ Roblox

**Core idea:** C2R takes coarse 3D renderings (low-poly geometry with flat shading encoding scene layout, camera motion, and human trajectories) and synthesizes photorealistic urban crowd videos using a ControlNet-style conditioning module on Wan 2.1. Text prompts guide appearance, lighting, weather, and city identity. The key insight is that camera motion, spatial structure, and temporal consistency are trivially handled in 3D, while textures, materials, lighting, and diversity are best handled by learned generative models — C2R bridges this complementary relationship.

---

## 1. Architecture Overview

### Base Model
- **Wan 2.1 T2V (14B)** — Flow Matching Diffusion Transformer
- 3D causal VAE (Wan-VAE) with 4×16×16 spatiotemporal compression
- UMT5-XXL text encoder
- DiT blocks with self-attention → cross-attention → FFN

### C2R Additions
The paper describes a **ControlNet-Transformer** architecture inspired by PIXART-δ:

1. **Feature extractor: DINOv3 (FROZEN)** — Meta's self-supervised ViT foundation model (arXiv 2508.10104, Aug 2025; 7B params trained on 1.7B unlabeled images). Both coarse CG renderings and real video frames are passed through a frozen DINOv3 encoder to extract dense spatio-temporal features. DINOv3's self-supervised features are naturally domain-invariant — this is THE mechanism that enables CG-to-real transfer without paired data.
2. **Control adapter:** A learned module that converts DINOv3 spatio-temporal features into an additive latent guidance signal for the DiT blocks. This is the ControlNet-Transformer component that gets trained.
3. **ControlNet blocks:** First N transformer blocks are duplicated; the ControlNet copy processes DINOv3-derived conditioning features, and its output is added to the corresponding main block output via zero-initialized projectors.
4. **Two-phase training strategy:**
   - **Phase 1 (Generative Distribution Alignment):** The Wan 2.1 diffusion backbone is adapted to real-world urban video while the VAE encoder remains frozen. Builds the photorealistic generation prior.
   - **Phase 2 (Spatio-Temporal Control Grounding):** The diffusion backbone is FROZEN, and the control adapter is trained to convert DINOv3 spatio-temporal features into additive latent guidance. Uses mixed CG-real data.

### Key Design: DINOv3 as the Shared Feature Space
The paper's "shared implicit spatio-temporal features" are **DINOv3 features** (Figure 2 caption explicitly states: _"a control adapter is trained to convert DINOv3 spatio-temporal features into an additive latent guidance signal"_). This is the critical architectural insight:

- DINOv3 produces semantically meaningful dense features regardless of visual domain (CG, game engine, photorealistic)
- A coarse box-geometry rendering and a real photo of the same scene produce similar DINOv3 features in terms of spatial layout/structure
- **No custom encoder needed** — frozen DINOv3 IS the shared encoder
- The control adapter only needs to learn: DINOv3 feature space → DiT latent space
- This is why C2R generalizes to arbitrary CG inputs (game footage, different 3D engines) without domain-specific fine-tuning

### Conditioning Strength Control
The paper describes a configurable conditioning strength parameter (similar to ControlNet's scale), allowing users to dial between "strong structural adherence" and "more creative freedom." This enables coarse-to-fine inputs — very low-poly inputs get more inpainting, while mid-coarse inputs are followed more faithfully.

---

## 2. Repository & Codebase Status

### Official Repo
- **GitHub:** https://github.com/GonzaloGNogales/coarse2real/
- **Status:** PLACEHOLDER ONLY — `README.md` + `LICENSE` (CC BY-NC-ND 4.0), no code released
- Code is marked "coming soon" with a planned structure of `configs/`, `data/`, `models/`, `scripts/`, `utils/`

### Author Background (Relevant Prior Code)

| Author | Affiliation | Relevant repos / expertise |
|--------|-------------|---------------------------|
| Gonzalo Gomez-Nogales | URJC (PhD student) | Physics simulation, ML, crowds, generative AI. First major paper — no prior open-source repos |
| Dan Casas | URJC / Amazon | `dancasas/SMPLitex` (SMPL texture estimation), CrowdDNA project (crowd dynamics as active matter), garment simulation. Strong 3D human + crowd expertise |
| Marc Comino-Trinidad | URJC | Co-author on SMPLitex with Casas |
| Yicong Hong | Adobe Research | Vision-language, embodied AI |
| Chongjian Ge | Adobe Research | Co-author on PixArt-α (!!!) — this is directly relevant |
| Yi Zhou | Roblox | Applications to game content (Roblox demos in the paper) |

**Critical finding:** Chongjian Ge is a co-author on **PixArt-α**, and the C2R ControlNet architecture is explicitly described as inspired by **PIXART-δ's ControlNet-Transformer**. This makes the PixArt-alpha codebase the most likely architectural starting point.

### Key Starting Point Repos

| Repo | Purpose | Stars | URL |
|------|---------|-------|-----|
| **Wan-Video/Wan2.1** | Base video generation model | High | https://github.com/Wan-Video/Wan2.1 |
| **facebookresearch/dinov3** | Frozen feature extractor — THE shared feature space | High | https://github.com/facebookresearch/dinov3 |
| **PixArt-alpha/PixArt-alpha** | ControlNet-Transformer reference architecture | 5K+ | https://github.com/PixArt-alpha/PixArt-alpha |
| **shalfun/WanControl** | Wan2.1 + ControlNet training (uses DiffSynth-Studio) | ~200 | https://github.com/shalfun/WanControl |
| **TheDenk/wan2.1-dilated-controlnet** | Dilated ControlNet for Wan2.1 (training + inference) | ~300 | https://github.com/TheDenk/wan2.1-dilated-controlnet |
| **modelscope/DiffSynth-Studio** | Training framework used by WanControl, supports Wan ControlNet | 11K+ | https://github.com/modelscope/DiffSynth-Studio |
| **TheDenk/wan2.2-controlnet** | Updated ControlNet for Wan2.2 | ~150 | https://github.com/TheDenk/wan2.2-controlnet |

---

## 3. Recommended Implementation Strategy

### Strategy: Build on WanControl + DiffSynth-Studio

The cleanest path is:

1. **Use `shalfun/WanControl`** as the primary codebase. It already implements ControlNet-Transformer for Wan2.1 using the PIXART-δ architecture and provides a working training pipeline via DiffSynth-Studio.
2. **Reference `TheDenk/wan2.1-dilated-controlnet`** for an alternative ControlNet architecture with dilated/strided blocks and pre-trained checkpoints for canny/depth/HED.
3. **Study the PIXART-δ ControlNet-Transformer implementation** in `PixArt-alpha/PixArt-alpha` for the canonical reference of how ControlNet adapts to DiT architectures (the first N blocks are cloned, outputs are added to main block outputs).
4. **The C2R-specific novelty** is: (a) using frozen DINOv3 as a domain-invariant feature extractor for both CG and real inputs, (b) a learned control adapter mapping DINOv3 features → DiT guidance, (c) the two-phase training strategy (backbone adaptation then frozen-backbone control grounding), (d) dataset curation from 5 continents.

### What Exists vs. What You Need to Build

| Component | Exists in | What to build |
|-----------|-----------|---------------|
| Wan 2.1 base model + weights | Wan-Video/Wan2.1, HuggingFace | Nothing — use as-is |
| Wan VAE (3D causal, 4×16×16) | Wan-Video/Wan2.1 | Nothing — use as-is |
| **DINOv3 feature extractor** | **Meta facebookresearch/dinov3** | **Nothing — use FROZEN as-is. This IS the shared feature space.** |
| ControlNet-Transformer architecture | WanControl, PixArt-δ | Adapt to accept DINOv3 features as conditioning input |
| Control adapter (DINOv3 → DiT) | Does NOT exist | Learned projection from DINOv3 feature space to DiT latent guidance |
| ControlNet training loop | WanControl / DiffSynth-Studio | Modify for two-phase strategy (Phase 1: backbone adapt, Phase 2: frozen backbone + train adapter) |
| Coarse rendering dataset | Does NOT exist | Full pipeline: 3D simulation → render → pair with text |
| Real urban video dataset | Partially exists (public footage) | Curation + annotation from 5 continents |
| Two-phase training scheduler | Does NOT exist | Phase 1 (generative alignment) → Phase 2 (control grounding) |
| Conditioning strength control | Partially in ControlNet scale | Expose as user-facing parameter |
| Text prompt conditioning | Wan 2.1 base (UMT5-XXL) | Nothing — use as-is |

---

## 4. Dataset Construction Plan

### Phase 1: Real Urban Videos
The paper describes curating footage from **five continents** covering diverse cities, weather conditions, lighting, and clothing styles. From the project page examples: Zurich, Paris, Nairobi, London, Shanghai, Tokyo, New York.

**Sources to consider:**
- YouTube Creative Commons urban footage
- Mapillary / Google Street View temporal sequences
- Academic driving datasets (nuScenes, KITTI, Waymo) — though these are car-centric
- Custom captures

**Annotation pipeline:**
- Dense text captions (city, weather, time of day, clothing, landmarks)
- Can use VLMs (Qwen2-VL, InternVL) for auto-captioning
- Manual curation for quality

### Phase 2: Coarse CG Paired Data
**Synthetic pipeline:**
1. Use a crowd simulation tool (MassMotion, Menge, or custom) to generate pedestrian trajectories
2. Render coarse 3D scenes using simple geometry (boxes for buildings, capsules for humans, flat ground planes)
3. Render from the same camera as real footage to create pseudo-pairs
4. For true paired data: render both coarse and fine versions of the same CG scene

**Tools for coarse rendering:**
- Blender (Python scripted) with minimal materials
- Unreal Engine with LOD0 / proxy meshes
- Custom OpenGL/Vulkan renderer

---

## 5. Phased Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
**Goal:** Verify base model works, set up training infrastructure

- [ ] Clone Wan-Video/Wan2.1, download 14B T2V weights
- [ ] Clone shalfun/WanControl and verify ControlNet training works on a toy example (e.g., canny edge → video)
- [ ] Study DiffSynth-Studio training pipeline, understand data format
- [ ] Set up multi-GPU training infrastructure
- **Validation:** Generate a video from text-only with Wan 2.1. Generate a ControlNet-conditioned video with WanControl using canny edges.

### Phase 2: DINOv3 Integration + Control Adapter (Week 2-3)
**Goal:** Integrate frozen DINOv3 as the conditioning feature extractor and build the control adapter

- [ ] Download DINOv3 pre-trained weights (ViT-L or ViT-H+ distilled variant for memory efficiency)
- [ ] Implement DINOv3 feature extraction pipeline: video frames → frozen DINOv3 → dense spatio-temporal features
- [ ] Build the control adapter: learned projection from DINOv3 feature space → ControlNet input dimensions
- [ ] Modify ControlNet input pipeline to accept DINOv3 features instead of edge maps
- [ ] Test with synthetic paired data: render a simple 3D scene, extract DINOv3 features from both coarse and real
- [ ] Verify gradient flow: DINOv3 frozen (no grad), adapter + ControlNet blocks trainable
- **Validation:** Forward pass completes. DINOv3 features from CG and real inputs show structural similarity. Loss decreases on toy paired data.

### Phase 3: Dataset Pipeline (Week 3-5)
**Goal:** Build the full data curation and preprocessing pipeline

- [ ] Collect and curate real urban videos (start with 1K, scale to 10K+)
- [ ] Build auto-captioning pipeline with VLM
- [ ] Build coarse rendering generation pipeline (Blender script or UE proxy)
- [ ] Generate synthetic paired coarse↔fine data (start with 100 pairs, scale to 1K+)
- [ ] Implement dataset class compatible with DiffSynth-Studio format
- **Validation:** Dataset loads correctly. Visual inspection of coarse↔real pairs.

### Phase 4: Two-Phase Training (Week 5-8)
**Goal:** Implement and run the full C2R training strategy

- [ ] **Stage 1 (Generative Distribution Alignment):** Adapt Wan 2.1 backbone to real urban video data, VAE frozen
- [ ] **Stage 2 (Spatio-Temporal Control Grounding):** Freeze backbone, train control adapter on mixed CG-real data with DINOv3 features
- [ ] Implement mixed batch sampling (configurable CG:real ratio)
- [ ] Monitor for CG artifact contamination in outputs
- **Validation:** Stage 1 — model generates realistic urban videos from text. Stage 2 — model follows coarse rendering structure.

### Phase 5: Inference & Evaluation (Week 8-10)
**Goal:** Full inference pipeline with conditioning strength control

- [ ] Inference script: coarse rendering + text prompt → realistic video
- [ ] Conditioning strength slider
- [ ] Evaluation metrics: FID, FVD, temporal consistency
- [ ] Test generalization: feed game footage (Roblox, UE low-poly) as coarse input
- **Validation:** Visual quality comparable to paper results. Generalization to unseen CG inputs.

---

## 6. Multi-Agent System for Claude Code

The following agents are designed to be placed in `.claude/agents/` and invoked during implementation.

### Agent Inventory

| Agent | Role | Model | Tools |
|-------|------|-------|-------|
| `paper-oracle` | Ground truth validator — knows paper specs | Opus | Read only |
| `architecture` | ControlNet-Transformer + conditioning module | Sonnet/Opus | Read, Write, Bash |
| `data-pipeline` | Dataset curation, coarse rendering, captioning | Sonnet | Read, Write, Bash |
| `training` | Two-phase training loop, mixed batching | Sonnet | Read, Write, Bash |
| `inference` | Inference pipeline + conditioning control | Sonnet | Read, Write, Bash |
| `integration` | Wan2.1 + WanControl + DiffSynth setup | Sonnet | Read, Write, Bash |
| `validator` | Checks code against paper + runs tests | Opus | Read, Bash |

### Usage Pattern

```
# Set up the base codebase
> Use the integration agent to set up Wan2.1 and WanControl

# Implement the conditioning module
> Use the architecture agent to implement the DINOv3 feature extractor integration and control adapter
> Use the paper-oracle to verify the architecture matches Section 3.2

# Build the dataset
> Use the data-pipeline agent to build the coarse rendering generator

# Train
> Use the training agent to implement the two-phase strategy
> Use the validator to check training matches the paper procedure

# Iterate
> /impl-loop paper=./paper.pdf section="Section 3" output=src/models/
```

---

## 7. Key Technical Details from the Paper

### ControlNet-Transformer Design (from PIXART-δ)
The ControlNet is applied to the **first N blocks** of the DiT. For each of these blocks:
- The ControlNet block is a clone of the main block (same architecture)
- It processes the conditioning signal (coarse rendering features)
- Its output is added to the main block's output before passing to the next block
- The ControlNet's last projection layer is zero-initialized so it contributes nothing at start

This is different from UNet-based ControlNet which uses skip connections between encoder and decoder. In a Transformer, there's no encoder/decoder split, so the additive approach on early blocks is the correct adaptation.

### Two-Phase Mixed CG-Real Training
- **Phase 1:** Real videos only. Model learns photorealistic generation prior. The conditioning path exists but may receive null/random input (or this phase is pure text-to-video fine-tuning).
- **Phase 2:** Mix of real and CG data. Small fraction of paired CG data teaches coarse→real mapping. Majority real data prevents CG artifacts from leaking in.

The paper compares against off-the-shelf ControlNet for Wan 2.1 and shows their approach yields more expressive results — suggesting the two-phase strategy and shared feature space are critical differentiators.

### Conditioning Strength
The ControlNet output is scaled by a factor `s ∈ [0, 1]` before being added to the main block. Lower `s` means more creative freedom (model ignores coarse input), higher `s` means stricter structural adherence.

---

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Paper omits critical architecture details | High (no code released) | Cross-reference PIXART-δ and WanControl implementations |
| Coarse rendering encoder design unclear | Medium | Start with simple VAE encoder, iterate based on results |
| Dataset scale insufficient for quality | High | Leverage large-scale pre-trained Wan 2.1 prior; C2R adds relatively little |
| CG artifacts leak into real generation | Medium | Careful CG:real ratio tuning; monitor with FID |
| Memory/compute for 14B model training | High | Start with 1.3B variant for prototyping; use gradient checkpointing |
| Two-phase training instability | Medium | Careful learning rate scheduling between phases |

---

## 9. Quick Start Recommendation

**For fastest time to results**, start with:

1. `TheDenk/wan2.1-dilated-controlnet` — it has pre-trained depth/canny ControlNets and training code
2. Download frozen DINOv3 (ViT-L distilled) and replace the depth/canny preprocessor with DINOv3 feature extraction + a lightweight control adapter
3. Train the adapter on a small dataset of coarse↔real pairs
4. This gives you a working baseline in ~1 week before investing in the full two-phase strategy

**For maximum fidelity to the paper**, use:

1. `shalfun/WanControl` (built on DiffSynth-Studio, closer to PIXART-δ architecture)
2. Integrate frozen DINOv3 + implement the full control adapter
3. Implement the full two-phase training (backbone adaptation → frozen backbone + adapter training)
4. Build the complete dataset pipeline
5. This is a 6-10 week project depending on compute availability
