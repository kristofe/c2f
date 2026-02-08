# C2R: Coarse-to-Real — Implementation Project

## Paper
"Coarse-to-Real: Generative Rendering for Populated Dynamic Scenes" (arXiv 2601.22301)
ControlNet-Transformer conditioned on coarse 3D renderings, built on Wan 2.1 T2V.

## Key Architecture
- Base: Wan 2.1 T2V (start with 1.3B for prototyping, 14B for production)
- **Feature extractor: Frozen DINOv3** (Meta, arXiv 2508.10104) — self-supervised ViT providing domain-invariant features for BOTH CG and real inputs. This IS the shared feature space.
- **Learned control adapter:** Maps DINOv3 features → DiT latent guidance signal
- ControlNet-Transformer (PIXART-δ style): first N blocks cloned, additive fusion, zero-init projector
- Two-phase training: Phase 1 = backbone adaptation on real video (no ControlNet), Phase 2 = freeze backbone, train adapter + ControlNet on mixed CG-real data with DINOv3 features

## Starting Codebases
- `Wan-Video/Wan2.1` — base model
- `facebookresearch/dinov3` — frozen feature extractor (THE shared feature space)
- `shalfun/WanControl` — ControlNet training on Wan (DiffSynth-Studio backend)
- `TheDenk/wan2.1-dilated-controlnet` — alternative ControlNet with pre-trained weights
- `PixArt-alpha/PixArt-alpha` — ControlNet-Transformer reference

## Agent System
See `agents/` for specialized agents. Key agents:
- `c2r-paper-oracle` — validates against paper specs
- `c2r-architecture` — model code
- `c2r-training` — two-phase training
- `c2r-data-pipeline` — dataset construction
- `c2r-integration` — environment setup
- `c2r-simplicity` — ensures code is simple, doesn't include fallbacks or handles errors 
- `c2r-purity` — ensures code is adheres stricly to the paper 

## Implementation Order
1. Environment setup (integration agent)
2. Verify base Wan 2.1 works
3. Download + integrate frozen DINOv3, verify feature extraction on CG vs real inputs
4. Implement control adapter + ControlNet blocks (architecture agent)
5. Build dataset pipeline (data agent)
6. Phase 1: adapt backbone on real urban video (training agent)
7. Phase 2: freeze backbone, train adapter + ControlNet on mixed CG-real with DINOv3 features (training agent)
8. Inference pipeline (inference agent)
9. Validate everything (validator agent + paper oracle)
