import sys
from typing import List, Optional
from pathlib import Path

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers.training_utils import set_seed

sys.path.append(".")
sys.path.append("..")

from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig, Range
from utils import latent_utils
from utils.latent_utils import load_latents_or_invert_images, load_latents_or_invert_multi_style
from utils.style_matcher import StyleMatcher


@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)


def run(cfg: RunConfig) -> List[Image.Image]:
    """Run appearance transfer with single or multi-style support."""
    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))
    set_seed(cfg.seed)
    
    if cfg.user_query is not None and cfg.style_base_dir is not None:
        matcher = StyleMatcher(cfg.style_base_dir)
        matches = matcher.match(cfg.user_query, top_k=cfg.top_k_styles)
        assert matches, f"No styles matched for query: {cfg.user_query}"
        cfg.style_image_paths = [match[1] for match in matches]
        cfg.ensure_style_weights_initialized()  # Initialize style_weights after setting style_image_paths
        print(f"Matched {len(matches)} styles from query: {cfg.user_query}")
        for i, (style_id, path, score) in enumerate(matches):
            print(f"  Style {i+1}: ID={style_id}, path={path}, score={score:.3f}")
    
    model = AppearanceTransferModel(cfg)
    
    if cfg.style_image_paths is not None:
        style_latents_list, style_noise_list = load_latents_or_invert_multi_style(model=model, cfg=cfg)
        model.set_multi_style_latents(style_latents_list)
        model.set_multi_style_noise(style_noise_list)
    else:
        latents_app, latents_struct, noise_app, noise_struct = load_latents_or_invert_images(model=model, cfg=cfg)
        model.set_latents(latents_app, latents_struct)
        model.set_noise(noise_app, noise_struct)
    
    print("Running appearance transfer...")
    images = run_appearance_transfer(model=model, cfg=cfg)
    print("Done.")
    return images


def run_appearance_transfer(model: AppearanceTransferModel, cfg: RunConfig) -> List[Image.Image]:
    """Run appearance transfer with single or multi-style support."""
    init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
    if cfg.style_image_paths is not None:
        expected_batch_size = 1 + len(cfg.style_image_paths)
        assert init_latents.shape[0] == expected_batch_size, \
            f"init_latents batch size {init_latents.shape[0]} != expected {expected_batch_size}"
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)
    
    if cfg.style_image_paths is not None:
        num_styles = len(cfg.style_image_paths)
        style_prompts = [""] * num_styles
        prompts = [cfg.prompt] + style_prompts
        assert len(prompts) == 1 + num_styles, \
            f"prompts length {len(prompts)} != expected {1 + num_styles} (output + {num_styles} styles)"
    else:
        prompts = [cfg.prompt, cfg.prompt_app, cfg.prompt_struct]
    
    images = model.pipe(
        prompt=prompts,
        latents=init_latents,
        guidance_scale=cfg.CFG,
        num_inference_steps=cfg.num_timesteps,
        swap_guidance_scale=cfg.swap_guidance_scale,
        callback=model.get_adain_callback(),
        eta=1,
        zs=init_zs,
        generator=torch.Generator('cuda').manual_seed(cfg.seed),
        cross_image_attention_range=Range(start=start_step, end=end_step),
        sparse_weight=cfg.sparse_weight,
        clip_weight=cfg.clip_weight,
        run_config=cfg
    ).images
    
    if cfg.style_image_paths is not None:
        images[0].save(cfg.output_path / f"out_transfer---seed_{cfg.seed}.png")
        for i in range(len(cfg.style_image_paths)):
            if i+1 < len(images):
                images[i+1].save(cfg.output_path / f"out_style_{i}---seed_{cfg.seed}.png")
        joined_images = np.concatenate(images[::-1], axis=1)
        Image.fromarray(joined_images).save(cfg.output_path / f"out_joined---seed_{cfg.seed}.png")
    else:
        images[0].save(cfg.output_path / f"out_transfer---seed_{cfg.seed}.png")
        images[1].save(cfg.output_path / f"out_style---seed_{cfg.seed}.png")
        images[2].save(cfg.output_path / f"out_struct---seed_{cfg.seed}.png")
        joined_images = np.concatenate(images[::-1], axis=1)
        Image.fromarray(joined_images).save(cfg.output_path / f"out_joined---seed_{cfg.seed}.png")
    return images


if __name__ == '__main__':
    main()
