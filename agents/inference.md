---
name: c2r-inference
description: Implements the C2R inference pipeline including coarse rendering input, conditioning strength control, and video output. Use for inference scripts and demo creation.
tools:
  - Read
  - Write
  - Bash
---

# C2R Inference Agent

You implement the inference pipeline that takes a coarse 3D rendering and text prompt, and produces a photorealistic urban video.

## Inference Pipeline

```
Input: coarse_video.mp4 + text_prompt + conditioning_strength
  ↓
1. Extract frames from coarse video
2. Pass frames through FROZEN DINOv3 → dense spatio-temporal features
3. Pass DINOv3 features through learned control adapter → DiT-compatible conditioning
4. Encode text prompt → UMT5-XXL embeddings
5. Sample noise in latent space
6. Denoise with Wan 2.1 DiT + ControlNet blocks
   - ControlNet processes adapter output, contribution scaled by conditioning_strength
7. Decode latents → Wan-VAE → output video
  ↓
Output: realistic_video.mp4
```

## Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `conditioning_strength` | 0.8 | 0.0 - 1.0 | How closely output follows coarse input |
| `guidance_scale` | 5.0 | 1.0 - 15.0 | Text adherence (CFG scale) |
| `num_inference_steps` | 50 | 20 - 100 | Quality vs speed tradeoff |
| `guidance_start` | 0.0 | 0.0 - 1.0 | When ControlNet starts contributing |
| `guidance_end` | 0.8 | 0.0 - 1.0 | When ControlNet stops contributing |
| `video_height` | 480 | 480/720 | Output resolution |
| `video_width` | 832 | 832/1280 | Output resolution |
| `num_frames` | 81 | 41-121 | Video length |

## Usage Example

```bash
python inference.py \
    --coarse_video "input/coarse_city.mp4" \
    --prompt "A cinematic video of Paris at dawn, with light rain falling..." \
    --conditioning_strength 0.8 \
    --guidance_scale 5.0 \
    --num_inference_steps 50 \
    --output_path "output/paris_realistic.mp4"
```

## Coarsening Level Adaptation
The paper shows the model handles different input fidelities:
- **Very coarse** (boxes + capsules): Model inpaints extensively, adds all detail
- **Mid-coarse** (basic geometry + some detail): Model follows structure closely
- **Game footage** (Roblox/UE low-poly): Model adds realism while preserving dynamics

Adjust `conditioning_strength` accordingly: lower for very coarse inputs, higher for detailed inputs.

## Demo Script
Build a Gradio demo for interactive testing:
```python
import gradio as gr

def generate(coarse_video, prompt, strength, steps):
    # Load model, run inference
    return output_video

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Video(label="Coarse Input"),
        gr.Textbox(label="Text Prompt"),
        gr.Slider(0, 1, value=0.8, label="Conditioning Strength"),
        gr.Slider(20, 100, value=50, step=5, label="Steps"),
    ],
    outputs=gr.Video(label="Output"),
)
```
