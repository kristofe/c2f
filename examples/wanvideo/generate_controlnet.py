import torch
import argparse
from tqdm import tqdm
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d


def model_fn_wan_video_control(
    dit: WanModel,
    x: torch.Tensor,
    x_c: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature=None,
    y=None,
    **kwargs,
):
    """Inference model function with ControlNet support.

    Based on model_fn_wan_video but adds control block processing,
    mirroring WanModel.forward() (wan_video_dit.py:366-401).
    """
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)
        clip_embedding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embedding, context], dim=1)

    x, (f, h, w) = dit.patchify(x)
    x_c, (_, _, _) = dit.patchify(x_c)

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    for idx, block in enumerate(dit.blocks):
        x = block(x, context, t_mod, freqs)
        if idx < dit.control_layers:
            x_c = dit.control_blocks[idx](x_c, context, t_mod, freqs)
            x = x + dit.control_blocks[idx].zero_linear(x_c)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video using trained ControlNet checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to Lightning .ckpt file")
    parser.add_argument("--dit_path", type=str, default="Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--text_encoder_path", type=str, default="Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_path", type=str, default="Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    parser.add_argument("--control_video", type=str, required=True, help="Path to control signal video (e.g., depth map)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--output_path", type=str, default="output.mp4")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--control_layers", type=int, default=15)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, nargs=2, default=[30, 52])
    parser.add_argument("--tile_stride", type=int, nargs=2, default=[15, 26])
    return parser.parse_args()


def main():
    args = parse_args()

    # Load base models
    print("Loading base models...")
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models(
        [args.dit_path, args.text_encoder_path, args.vae_path],
        control_layers=args.control_layers,
    )

    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")

    # Load checkpoint (contains only control_blocks.* parameters)
    print(f"Loading checkpoint: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
    pipe.dit.load_state_dict(ckpt, strict=False)
    print(f"  Loaded {len(ckpt)} parameter tensors from checkpoint")

    # Ensure model dimensions match training resolution
    height = args.height // 16 * 16
    width = args.width // 16 * 16
    num_frames = args.num_frames
    if num_frames % 4 != 1:
        num_frames = (num_frames + 2) // 4 * 4 + 1
        print(f"  Adjusted num_frames to {num_frames} (must satisfy num_frames % 4 == 1)")

    tiler_kwargs = {
        "tiled": args.tiled,
        "tile_size": tuple(args.tile_size),
        "tile_stride": tuple(args.tile_stride),
    }

    # Encode control video through VAE
    print(f"Encoding control video: {args.control_video}")
    control_video = VideoData(args.control_video, height=height, width=width)
    pipe.vae.to(pipe.device)
    control_frames = pipe.preprocess_images([control_video[i] for i in range(num_frames)])
    control_tensor = torch.stack(control_frames, dim=2).to(dtype=pipe.torch_dtype, device=pipe.device)
    control_latents = pipe.encode_video(control_tensor, **tiler_kwargs).to(dtype=pipe.torch_dtype, device=pipe.device)
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    # Encode text prompt
    print("Encoding text prompt...")
    pipe.text_encoder.to(pipe.device)
    prompt_emb_posi = pipe.encode_prompt(args.prompt, positive=True)
    if args.cfg_scale != 1.0:
        prompt_emb_nega = pipe.encode_prompt(args.negative_prompt, positive=False)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # Initialize noise
    noise_shape = (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8)
    noise = pipe.generate_noise(noise_shape, seed=args.seed, device="cpu", dtype=torch.float32)
    latents = noise.to(dtype=pipe.torch_dtype, device=pipe.device)

    # Set up scheduler
    pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=args.sigma_shift)

    # Denoising loop with control
    print(f"Denoising ({args.num_inference_steps} steps)...")
    pipe.dit.to(pipe.device)
    with torch.no_grad():
        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            noise_pred_posi = model_fn_wan_video_control(
                pipe.dit, latents, control_latents,
                timestep=timestep, **prompt_emb_posi,
            )
            if args.cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video_control(
                    pipe.dit, latents, control_latents,
                    timestep=timestep, **prompt_emb_nega,
                )
                noise_pred = noise_pred_nega + args.cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], latents)

    # Decode to video frames
    print("Decoding video...")
    pipe.dit.to("cpu")
    torch.cuda.empty_cache()
    pipe.vae.to(pipe.device)
    frames = pipe.decode_video(latents, **tiler_kwargs)
    frames = pipe.tensor2video(frames[0])

    # Save
    save_video(frames, args.output_path, fps=15, quality=5)
    print(f"Video saved to: {args.output_path}")


if __name__ == "__main__":
    main()


'''

python examples/wanvideo/generate_controlnet.py  \
  --checkpoint_path lightning_logs/version_0/checkpoints/epoch=3-step=1000.ckpt \
  --dit_path Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
  --text_encoder_path Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth \
  --vae_path Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
  --control_video data_r10k/train/video_00001_c.mp4 \
  --prompt "walking into a dining room with shag rugs and polka dot walls." \
  --output_path controlnet_test.mp4 \
  --tiled \
  --cfg_scale 5

'''