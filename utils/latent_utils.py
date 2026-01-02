from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig
from utils import image_utils
from utils.ddpm_inversion import invert

def load_latents_or_invert_images(model: AppearanceTransferModel, cfg: RunConfig):
    """Load or invert images for single or multi-style transfer."""
    if cfg.style_image_paths is not None:
        # Multi-style mode
        return load_latents_or_invert_multi_style(model, cfg)
    else:
        # Single style mode (original)
        print(cfg.app_latent_save_path)
        if cfg.load_latents and cfg.app_latent_save_path.exists() and cfg.struct_latent_save_path.exists():
            print("Loading existing latents...")
            latents_app, latents_struct = load_latents(cfg.app_latent_save_path, cfg.struct_latent_save_path)
            noise_app, noise_struct = load_noise(cfg.app_latent_save_path, cfg.struct_latent_save_path)
            print("Done.")
        else:
            print("Inverting images...")
            app_image, struct_image = image_utils.load_images(cfg=cfg, save_path=cfg.output_path)
            model.enable_edit = False
            latents_app, latents_struct, noise_app, noise_struct = invert_images(
                app_image=app_image,
                struct_image=struct_image,
                sd_model=model.pipe,
                cfg=cfg
            )
            model.enable_edit = True
            print("Done.")
        return latents_app, latents_struct, noise_app, noise_struct


def load_latents_or_invert_multi_style(model: AppearanceTransferModel, cfg: RunConfig):
    """Load or invert multiple style images. All styles are treated equally, no separate struct."""
    assert cfg.style_image_paths is not None and len(cfg.style_image_paths) > 0
    
    style_latents_list = []
    style_noise_list = []
    
    for i, style_path in enumerate(cfg.style_image_paths):
        style_latent_path = cfg.latents_path / f"style_{i}_{style_path.stem}.pt"
        style_noise_path = cfg.latents_path / f"style_{i}_{style_path.stem}_ddpm_noise.pt"
        
        if cfg.load_latents and style_latent_path.exists() and style_noise_path.exists():
            print(f"Loading existing latents for style {i}...")
            latents = torch.load(style_latent_path)
            noise = torch.load(style_noise_path)
            if isinstance(latents, list):
                latents = [l.to("cuda") for l in latents]
            else:
                latents = latents.to("cuda")
            noise = noise.to("cuda")
        else:
            print(f"Inverting style image {i}...")
            style_image = image_utils.load_size(style_path)
            model.enable_edit = False
            noise, latents = invert(
                x0=torch.from_numpy(np.array(style_image)).float().permute(2, 0, 1).unsqueeze(0).to('cuda') / 127.5 - 1.0,
                pipe=model.pipe,
                prompt_src=cfg.prompt_app if cfg.prompt_app else "",
                num_diffusion_steps=cfg.num_timesteps,
                cfg_scale_src=cfg.CFG
            )
            model.enable_edit = True
            torch.save(latents, style_latent_path)
            torch.save(noise, style_noise_path)
        
        style_latents_list.append(latents)
        style_noise_list.append(noise)
    
    return style_latents_list, style_noise_list


def load_latents(app_latent_save_path: Path, struct_latent_save_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    latents_app = torch.load(app_latent_save_path)
    latents_struct = torch.load(struct_latent_save_path)
    if type(latents_struct) == list:
        latents_app = [l.to("cuda") for l in latents_app]
        latents_struct = [l.to("cuda") for l in latents_struct]
    else:
        latents_app = latents_app.to("cuda")
        latents_struct = latents_struct.to("cuda")
    return latents_app, latents_struct


def load_noise(app_latent_save_path: Path, struct_latent_save_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    latents_app = torch.load(app_latent_save_path.parent / ('app_'+app_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_struct = torch.load(struct_latent_save_path.parent / ('struct_'+struct_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_app = latents_app.to("cuda")
    latents_struct = latents_struct.to("cuda")
    return latents_app, latents_struct


def invert_images(sd_model: AppearanceTransferModel, app_image: Image.Image, struct_image: Image.Image, cfg: RunConfig):
    input_app = torch.from_numpy(np.array(app_image)).float() / 127.5 - 1.0
    input_struct = torch.from_numpy(np.array(struct_image)).float() / 127.5 - 1.0
    if cfg.resize:
        input_app = crop_and_resize(input_app)
        input_struct = crop_and_resize(input_struct)
    zs_app, latents_app = invert(x0=input_app.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                 pipe=sd_model,
                                 prompt_src=cfg.prompt_app,
                                 num_diffusion_steps=cfg.num_timesteps,
                                 cfg_scale_src=cfg.CFG)
    # zs_struct, latents_struct = invert(x0=input_struct.permute(2, 0, 1).unsqueeze(0).to('cuda'),
    #                                     pipe=sd_model,
    #                                     prompt_src=cfg.prompt_struct,
    #                                     num_diffusion_steps=cfg.num_timesteps,
    #                                     cfg_scale_src=cfg.CFG)
    if cfg.app_image_path==cfg.struct_image_path:
        print("---------single style--------")
        zs_struct = zs_app
        latents_struct = latents_app
    else:
        print("---------multi style--------")
        zs_struct, latents_struct = invert(x0=input_struct.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                        pipe=sd_model,
                                        prompt_src=cfg.prompt_struct,
                                        num_diffusion_steps=cfg.num_timesteps,
                                        cfg_scale_src=cfg.CFG)
    
    # Save the inverted latents and noises
    torch.save(latents_app, cfg.latents_path / f"app_{cfg.app_image_path.stem}.pt")
    torch.save(latents_struct, cfg.latents_path / f"struct_{cfg.struct_image_path.stem}.pt")
    torch.save(zs_app, cfg.latents_path / f"app_{cfg.app_image_path.stem}_ddpm_noise.pt")
    torch.save(zs_struct, cfg.latents_path / f"struct_{cfg.struct_image_path.stem}_ddpm_noise.pt")
    return latents_app, latents_struct, zs_app, zs_struct


def get_init_latents_and_noises(model: AppearanceTransferModel, cfg: RunConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get initial latents and noises for single or multi-style transfer."""
    if cfg.style_image_paths is not None:
        return get_init_latents_and_noises_multi_style(model, cfg)
    
    # Single style mode (original)
    if model.latents_struct.dim() == 4 and model.latents_app.dim() == 4 and model.latents_app.shape[0] > 1:
        model.latents_struct = model.latents_struct[cfg.skip_steps]
        model.latents_app = model.latents_app[cfg.skip_steps]
        if cfg.skip_steps==-1:
            print("Initialize the latents.")
            z_T = torch.load("source_noise.pt")
        else:
            z_T = model.latents_struct

    init_latents = torch.stack([z_T, model.latents_app, model.latents_struct])
    init_zs = [model.zs_struct[cfg.skip_steps:], model.zs_app[cfg.skip_steps:], model.zs_struct[cfg.skip_steps:]]
    return init_latents, init_zs


def get_init_latents_and_noises_multi_style(model: AppearanceTransferModel, cfg: RunConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get initial latents and noises for multi-style transfer. Structure: [output, style1, style2, ..., styleN]"""
    assert hasattr(model, 'style_latents_list') and len(model.style_latents_list) > 0
    
    # Select latents based on skip_steps (same logic as original)
    style_latents_selected = []
    style_noise_selected = []
    
    for latents, noise in zip(model.style_latents_list, model.style_noise_list):
        if latents.dim() == 4 and latents.shape[0] > 1:
            latents = latents[cfg.skip_steps]
        style_latents_selected.append(latents)
        style_noise_selected.append(noise[cfg.skip_steps:] if hasattr(noise, '__getitem__') else noise)
    
    # Use first style as output initial latent
    if style_latents_selected[0].dim() == 4 and style_latents_selected[0].shape[0] > 1:
        z_T = style_latents_selected[0]
    else:
        if cfg.skip_steps == -1:
            print("Initialize the latents.")
            z_T = torch.load("source_noise.pt")
        else:
            z_T = style_latents_selected[0]
    
    # Stack: [output, style1, style2, ..., styleN]
    init_latents = torch.stack([z_T] + style_latents_selected)
    # Noise: [output_noise, style1_noise, style2_noise, ..., styleN_noise]
    output_noise = style_noise_selected[0][cfg.skip_steps:] if hasattr(style_noise_selected[0], '__getitem__') else style_noise_selected[0]
    init_zs = [output_noise] + style_noise_selected
    
    return init_latents, init_zs


def crop_and_resize(img, target_size=(512, 512), threshold=0.85):
    img_np = np.array(img)
    mask = img_np.mean(axis=2, keepdims=True) < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        print("all white")
        return img
    y0, x0, _ = coords.min(axis=0)
    y1, x1, _ = coords.max(axis=0) + 1
    if np.abs(x0-x1)*np.abs(y0-y1)>(256*256):
        return img
    cropped_img = img[y0:y1,x0:x1,:]#img.crop((x0, y0, x1, y1))
    cropped_img = cropped_img.permute(2,0,1)
    print(cropped_img.shape)
    pad_length = y1-y0 if y1-y0>x1-x0 else x1-x0
    pad_length = int(pad_length/2.5)

    rotat = transforms.Compose([transforms.RandomRotation(degrees=(-30, 30),fill=1),
                                transforms.RandomHorizontalFlip(0.5),
                                ])
    if np.abs(x0-x1)*np.abs(y0-y1)<(128*128):
        cropped_img = torch.cat([cropped_img, rotat(cropped_img)],dim=1)
        cropped_img = torch.cat([cropped_img, rotat(cropped_img)],dim=2)
    else:
        if np.abs(x0-x1)<np.abs(y0-y1):
            cropped_img = torch.cat([cropped_img, rotat(cropped_img)],dim=2)
        else:
            cropped_img = torch.cat([cropped_img, rotat(cropped_img)],dim=1)
            
    transform = transforms.Compose([
        transforms.Pad(pad_length,fill=1, padding_mode='constant'),
        transforms.Resize(target_size,transforms.InterpolationMode.BICUBIC)
    ])
    resized_img = transform(cropped_img)
    print('image is croped and resized')
    return resized_img.permute(1,2,0)