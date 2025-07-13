import torch
import decord
from utils.infer_utils.config import parse_args, prepare_inputs
from utils.infer_utils.inference_utils import inference

if __name__ == "__main__":
    decord.bridge.set_bridge("torch")
    args = parse_args(is_multi=False)
    out_name, latents_path, prompt_list, negative_prompt_list, lora_paths = prepare_inputs(args, is_multi=False)
    
    video_frames = inference(
        model=args.model,
        prompt=prompt_list,
        negative_prompt=negative_prompt_list,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        xformers=args.xformers,
        sdp=args.sdp,
        lora_path=lora_paths["lora_path"],
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
        seed=args.seed,
        latents_path=latents_path,
        noise_prior=args.noise_prior,
        repeat_num=args.repeat_num,
        fps=args.fps,
        out_name=out_name,
        is_multi=False
    )