"""
Example usage of multi-style transfer with style matching.
"""
from pathlib import Path
from config import RunConfig
from run import run


def example_style_matching():
    """Match styles from user query and run transfer."""
    cfg = RunConfig(
        output_path=Path("./output_final/style_matching/"),
        domain_name="a cat",
        seed=42,
        num_timesteps=100,
        CFG=15,
        swap_guidance_scale=3.5,
        user_query="clean sharp outlines, black and white block shading, unadorned line structure, layered visual effect, no redundant colors or strokes, simple and neat composition",
        style_base_dir=Path("./style_base"),
        top_k_styles=2,
        interpolation=0.1,
        sparse_weight=10,
        clip_weight=25
    )
    
    images = run(cfg)
    return images


def example_direct_styles():
    """Directly specify style images without matching."""
    cfg = RunConfig(
        output_path=Path("./output_final/style_assign_1~3-2"),
        domain_name="a cat",
        seed=42,
        num_timesteps=100,
        CFG=15,
        swap_guidance_scale=3.5,
        style_image_paths=[
            Path("./style_base/1.png"),
            Path("./style_base/2.png"),
            Path("./style_base/3.png"),
            #Path("./style_base/4.png"),
        ],
        style_weights=[0.5, 0.25, 0.25],
        interpolation=0.1,
        sparse_weight=10,
        clip_weight=25,
        per_style_interpolation=False # 超过3个风格别开 3个风格已经需要40G显存 作用是选择k v 的拼接方式 但是开了效果会变好
    )
    
    images = run(cfg)
    return images


if __name__ == "__main__":
    example_style_matching()
    #example_direct_styles()
