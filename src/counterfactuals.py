import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

from .utils import IM_SIZE, NUM_FRAMES, normalize_clip, uniform_frame_indices


def make_shuffle_counterfactual(clip: torch.Tensor, seed: int = 0):
    """
    clip: (1,3,T,H,W)
    Returns shuffled clip and the permutation.
    """
    rng = np.random.RandomState(seed)
    T = clip.shape[2]
    perm = rng.permutation(T)
    out = clip.clone()
    out[:, :, :, :, :] = clip[:, :, perm, :, :]
    return out, perm


def load_clip_with_custom_indices(video_path: str, idx: np.ndarray, device=None):
    """
    Decode frames at indices idx and return clip (1,3,T,112,112) normalized.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(idx).asnumpy()  # (T,H,W,3)

    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T,3,H,W)
    x = F.interpolate(x, size=(IM_SIZE, IM_SIZE), mode="bilinear", align_corners=False)

    x = normalize_clip(x)  # (T,3,112,112)
    x = x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # (1,3,T,112,112)

    if device is not None:
        x = x.to(device)
    return x


def make_speed_counterfactual(video_path: str, mode: str = "fast", device=None):
    """
    Creates counterfactual clips by changing temporal sampling density.
    - fast: more skipping
    - slow: sample from a narrower middle window
    Returns clip and indices used.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    n = len(vr)

    if mode == "fast":
        idx = uniform_frame_indices(n, NUM_FRAMES)
        idx = np.clip((idx * 1.2).astype(int), 0, n - 1)
    elif mode == "slow":
        center = n // 2
        half = min(n // 4, 30)
        start = max(0, center - half)
        end = min(n - 1, center + half)
        idx = np.linspace(start, end, NUM_FRAMES).astype(int)
    else:
        raise ValueError("mode must be 'fast' or 'slow'")

    clip = load_clip_with_custom_indices(video_path, idx, device=device)
    return clip, idx