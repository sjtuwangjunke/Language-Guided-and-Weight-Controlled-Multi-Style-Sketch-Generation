"""
Style matching module for finding top-k style images based on user query.
"""
from pathlib import Path
from typing import List, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


STYLE_DATABASE = {
    1: {
        "keywords": ["规整", "填色本", "干净轮廓", "利落", "无修饰", "regular", "coloring book", "clean outline", "neat", "unadorned"],
        "description": "A neat line art style with clean and sharp outlines, presenting the effect of coloring book patterns, without any extra decorations or modifications."
    },
    2: {
        "keywords": ["粗细变化", "轻重", "灵动", "起伏", "thick-thin variation", "light and heavy strokes", "lively", "undulating"],
        "description": "A vivid line art style with obvious changes in line thickness and stroke weight, showing natural undulations and making the overall lines look lively and dynamic."
    },
    3: {
        "keywords": ["阴影", "黑白", "层次", "重色点缀", "局部涂黑", "质感", "shadow", "black and white", "layered", "dark accents", "partial black filling", "textured"],
        "description": "A black-and-white line art style with block shadows and partial black filling, using dark accents to create rich layers and enhance the overall textured effect of the drawing."
    },
    4: {
        "keywords": ["铅笔", "随性", "草稿", "松散轮廓", "自然", "手绘", "轻描", "pencil", "casual", "draft", "loose outline", "natural", "hand-drawn", "light sketch"],
        "description": "A casual hand-drawn pencil sketch style with loose and natural outlines, presenting the relaxed and unconstrained feeling of a draft, characterized by light and gentle strokes."
    }
}


class StyleMatcher:
    """Match user query to top-k style images using semantic similarity."""
    
    def __init__(self, style_base_dir: Path, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.style_base_dir = Path(style_base_dir)
        assert self.style_base_dir.exists(), f"Style base dir not found: {style_base_dir}"
        
        # Use local cache if available to avoid network requests
        cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers" / model_name
        if cache_dir.exists() and (cache_dir / "config.json").exists():
            model_path = str(cache_dir)
            print(f"✓ Using local model from: {model_path}")
        else:
            model_path = model_name
            print(f"Loading model from Hugging Face: {model_name}")
        
        print(f"Loading style embeddings...")
        self.encoder = SentenceTransformer(model_path)
        self.style_embeddings = self._build_style_embeddings()
        self.style_image_paths = self._collect_style_images()
    
    def _build_style_embeddings(self) -> torch.Tensor:
        """Build embeddings for all style descriptions."""
        descriptions = []
        for style_id, style_info in STYLE_DATABASE.items():
            desc = f"{style_info['description']} {' '.join(style_info['keywords'])}"
            descriptions.append(desc)
        
        embeddings = self.encoder.encode(descriptions, convert_to_tensor=True)
        return embeddings
    
    def _collect_style_images(self) -> dict:
        """Collect style images from style_base directory."""
        image_paths = {}
        for style_id in STYLE_DATABASE.keys():
            img_path = self.style_base_dir / f"{style_id}.png"
            if img_path.exists():
                image_paths[style_id] = img_path
        return image_paths
    
    def _extract_style_keywords(self, query: str) -> str:
        """Extract style-related keywords from user query."""
        query_lower = query.lower()
        matched_keywords = []
        
        for style_info in STYLE_DATABASE.values():
            for keyword in style_info["keywords"]:
                if keyword.lower() in query_lower:
                    matched_keywords.append(keyword)
        
        if matched_keywords:
            return " ".join(matched_keywords)

        return query
    
    def match(self, query: str, top_k: int = 1) -> List[Tuple[int, Path, float]]:
        """Match user query to top-k style images."""
        assert top_k > 0, "top_k must be positive"
        assert top_k <= len(STYLE_DATABASE), f"top_k cannot exceed {len(STYLE_DATABASE)}"
        
        # Extract style keywords from query
        style_query = self._extract_style_keywords(query)
        
        # Check if we extracted keywords (not the full query)
        extracted_keywords = style_query != query
        
        # Calculate keyword match scores
        keyword_scores = np.zeros(len(STYLE_DATABASE))
        query_lower = query.lower()
        for style_id, style_info in STYLE_DATABASE.items():
            for keyword in style_info["keywords"]:
                if keyword.lower() in query_lower:
                    keyword_scores[style_id - 1] += 1.0
        
        # If we have keyword matches, prioritize them
        if extracted_keywords and keyword_scores.max() > 0:
            # Use keyword match as primary score
            scores = keyword_scores
        else:
            # No keyword match, use semantic similarity
            query_embedding = self.encoder.encode(style_query, convert_to_tensor=True)
            scores = cosine_similarity(
                query_embedding.cpu().numpy().reshape(1, -1),
                self.style_embeddings.cpu().numpy()
            )[0]
        
        # Get top-k style IDs
        top_style_indices = np.argsort(scores)[::-1][:top_k]
        top_style_ids = [idx + 1 for idx in top_style_indices]
        
        # Return matched styles
        results = []
        for style_id in top_style_ids:
            if style_id in self.style_image_paths:
                results.append((style_id, self.style_image_paths[style_id], float(scores[style_id - 1])))
        
        return results
