from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional, List

from numpy.random import weibull


class Range(NamedTuple):
    start: int
    end: int


@dataclass
class RunConfig:
    app_image_path: Optional[Path] = None
    struct_image_path: Optional[Path] = None
    domain_name: Optional[str] = None
    ref_name: Optional[str] = None
    struct_name: Optional[str] = None
    output_path: Path = Path('./output')
    seed: int = 42
    prompt: Optional[str] = None
    num_timesteps: int = 100
    use_masked_adain: bool = False
    cross_attn_64_range: Range = Range(start=0, end=90)
    cross_attn_32_range: Range = Range(start=0, end=70)
    adain_range: Range = Range(start=0, end=100)
    swap_guidance_scale: float = 3.5
    CFG: float = 3.5
    contrast_strength: float = 1.67
    object_noun: Optional[str] = None
    load_latents: bool = True
    skip_steps: int = 0
    sparse_weight: float = 10
    clip_weight: float = 25
    mix_style: bool = False
    interpolation: float = 0
    resize: bool = False
    alpha: float = 0.5
    style_image_paths: Optional[List[Path]] = None
    style_weights: Optional[List[float]] = None
    user_query: Optional[str] = None
    style_base_dir: Optional[Path] = Path('./style_base')
    top_k_styles: int = 1
    # If True, use per-style interpolation (old method: create separate interpolation for each style)
    # If False, use combined style interpolation (new method: fuse all styles first)
    per_style_interpolation: bool = False

    def __post_init__(self):
        if self.style_image_paths is None and self.user_query is None:
            assert self.app_image_path is not None, "app_image_path required when not using style matching"
            assert self.struct_image_path is not None, "struct_image_path required when not using style matching"
            save_name = f'app={self.app_image_path.stem}---struct={self.struct_image_path.stem}'
        else:
            save_name = "multi_style"
        
        self.output_path = self.output_path / (self.domain_name or "object") / save_name
        self.output_path.mkdir(parents=True, exist_ok=True)

        if self.use_masked_adain and self.domain_name is None:
            raise ValueError("Must provide domain_name and prompt when using masked AdaIN")
        if self.domain_name is None:
            self.domain_name = "object"
        if self.prompt is None:
            self.prompt = f"a sketch of {self.domain_name}"
            self.prompt_app = ""
            self.prompt_struct = ""
        if self.object_noun is None:
            self.object_noun = self.domain_name

        if self.style_image_paths is not None:
            assert len(self.style_image_paths) > 0, "style_image_paths must not be empty"
            if self.style_weights is not None:
                assert len(self.style_weights) == len(self.style_image_paths), \
                    "style_weights length must match style_image_paths length"
                assert all(w >= 0 for w in self.style_weights), "All style_weights must be non-negative"
                total_weight = sum(self.style_weights)
                assert total_weight > 0, "Sum of style_weights must be positive"
                self.style_weights = [w / total_weight for w in self.style_weights]
            else:
                self.style_weights = [1.0 / len(self.style_image_paths)] * len(self.style_image_paths)
        
        self.latents_path = Path(self.output_path) / "latents"
        self.latents_path.mkdir(parents=True, exist_ok=True)
        
        if self.app_image_path is not None:
            self.app_latent_save_path = self.latents_path / f"app_{self.app_image_path.stem}.pt"
        if self.struct_image_path is not None:
            self.struct_latent_save_path = self.latents_path / f"sturct_{self.struct_image_path.stem}.pt"
    
    def ensure_style_weights_initialized(self):
        """Ensure style_weights is initialized if style_image_paths is set."""
        if self.style_image_paths is not None and len(self.style_image_paths) > 0:
            if self.style_weights is None:
                weights = []
                for idx in range(len(self.style_image_paths)):
                    weights.append(0.5**(idx + 1))
                self.style_weights = [w / sum(weights) for w in weights]
                print (self.style_weights)
            else:
                assert len(self.style_weights) == len(self.style_image_paths), \
                    "style_weights length must match style_image_paths length"
                assert all(w >= 0 for w in self.style_weights), "All style_weights must be non-negative"
                total_weight = sum(self.style_weights)
                assert total_weight > 0, "Sum of style_weights must be positive"
                self.style_weights = [w / total_weight for w in self.style_weights]
