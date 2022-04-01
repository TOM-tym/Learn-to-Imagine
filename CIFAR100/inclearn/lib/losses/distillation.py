import functools
import math

import torch
from torch.nn import functional as F

def compute_collapsed_features(a, b, collapse_channels, normalize):
    assert a.shape == b.shape, (a.shape, b.shape)
    a = compute_collapsed_features_single(a, collapse_channels, normalize)
    b = compute_collapsed_features_single(b, collapse_channels, normalize)
    return a, b

def compute_collapsed_features_single(a, collapse_channels, normalize):
    a = torch.pow(a, 2)
    if collapse_channels == "channels":
        a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
    elif collapse_channels == "width":
        a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
    elif collapse_channels == "height":
        a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
    elif collapse_channels == "gap":
        a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
    elif collapse_channels == "spatial":
        a_h = a.sum(dim=3).view(a.shape[0], -1)
        a_w = a.sum(dim=2).view(a.shape[0], -1)
        a = torch.cat([a_h, a_w], dim=-1)
    else:
        raise ValueError("Unknown method to collapse: {}".format(collapse_channels))
    if normalize:
        a = F.normalize(a, dim=1, p=2)
    return a