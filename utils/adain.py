"""
AdaIN (Adaptive Instance Normalization) utilities for style transfer.
"""
import torch


def masked_adain(content_feat, style_feat, content_mask, style_mask):
    """Apply AdaIN with masks."""
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat, mask=style_mask)
    content_mean, content_std = calc_mean_std(content_feat, mask=content_mask)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    style_normalized_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    return content_feat * (1 - content_mask) + style_normalized_feat * content_mask


def adain(content_feat, style_feat):
    """Apply AdaIN without masks."""
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def multi_style_adain(content_feat, style_feats: list, weights: list):
    """
    Apply AdaIN with multiple style features weighted combination.
    
    Args:
        content_feat: Content feature tensor
        style_feats: List of style feature tensors
        weights: List of weights (will be normalized to sum to 1)
    
    Returns:
        Stylized feature tensor
    """
    assert len(style_feats) > 0, "At least one style feature required"
    assert len(style_feats) == len(weights), "Style features and weights must have same length"
    
    # Normalize weights
    weights = torch.tensor(weights, device=content_feat.device, dtype=content_feat.dtype)
    weights = weights / weights.sum()
    
    # Apply AdaIN for each style and combine
    stylized_feats = []
    for style_feat in style_feats:
        stylized_feats.append(adain(content_feat, style_feat))
    
    # Weighted combination
    result = sum(w * feat for w, feat in zip(weights, stylized_feats))
    return result


def calc_mean_std(feat, eps=1e-5, mask=None):
    """Calculate mean and std of feature tensor."""
    size = feat.size()
    if len(size) == 2:
        return calc_mean_std_2d(feat, eps, mask)

    assert (len(size) == 3)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)

    return feat_mean, feat_std


def calc_mean_std_2d(feat, eps=1e-5, mask=None):
    """Calculate mean and std of 2D feature tensor."""
    size = feat.size()
    assert (len(size) == 2)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1)

    return feat_mean, feat_std
