import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from decord import VideoReader, cpu

from .utils import IM_SIZE, NUM_FRAMES, LABEL_MAP, uniform_frame_indices, normalize_clip


class UCF101BinaryDataset(Dataset):
    """
    Loads UCF101 videos for a binary subset (Basketball vs BasketballDunk).
    Expects a dataframe-like object with columns: path, label (0/1).
    """

    def __init__(self, df, im_size: int = IM_SIZE, num_frames: int = NUM_FRAMES):
        self.df = df.reset_index(drop=True)
        self.im_size = im_size
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def _decode_uniform_frames(self, path: str) -> torch.Tensor:
        """
        Returns a clip tensor of shape (3, T, H, W), normalized.
        """
        vr = VideoReader(path, ctx=cpu(0))
        n = len(vr)
        idx = uniform_frame_indices(n, self.num_frames)
        frames = vr.get_batch(idx).asnumpy()  # (T,H,W,3) uint8

        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T,3,H,W)
        x = F.interpolate(x, size=(self.im_size, self.im_size), mode="bilinear", align_corners=False)

        x = normalize_clip(x)                 # (T,3,H,W) normalized
        x = x.permute(1, 0, 2, 3).contiguous()  # (3,T,H,W)
        return x

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        label = int(self.df.loc[idx, "label"])

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        clip = self._decode_uniform_frames(path)
        return clip, torch.tensor(label, dtype=torch.long)