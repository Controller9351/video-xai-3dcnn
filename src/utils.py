import numpy as np
import torch

IM_SIZE = 112
NUM_FRAMES = 16

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

LABEL_MAP = {"Basketball": 0, "BasketballDunk": 1}
INV_LABEL_MAP = {0: "Basketball", 1: "BasketballDunk"}

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_clip(x: torch.Tensor) -> torch.Tensor:
    """
    x: (T,3,H,W) in [0,1]
    returns normalized tensor (T,3,H,W)
    """
    mean = IMAGENET_MEAN.to(x.device)
    std = IMAGENET_STD.to(x.device)
    return (x - mean) / std

def uniform_frame_indices(n_frames: int, num_frames: int = NUM_FRAMES) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("Video has 0 frames.")
    if n_frames == 1:
        return np.zeros((num_frames,), dtype=int)
    return np.linspace(0, n_frames - 1, num_frames).astype(int)