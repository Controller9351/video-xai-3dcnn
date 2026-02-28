import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .integrated_gradients import integrated_gradients_video, temporal_importance_from_ig
from .gradcam import upsample_cam_to_input


@torch.no_grad()
def predict_probs(model, clip):
    """
    clip: (1,3,T,H,W)
    returns pred, probs (numpy)
    """
    model.eval()
    logits = model(clip)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    return pred, probs


@torch.no_grad()
def predict_prob_of_class(model, clip, class_idx: int):
    model.eval()
    logits = model(clip)
    probs = torch.softmax(logits, dim=1)[0]
    return float(probs[class_idx].item())


def occlude_frames_mean(clip_tensor, frame_ids):
    """
    Replace selected frames with per-frame mean (more natural than zeros).
    clip_tensor: (1,3,T,H,W)
    """
    clip_mod = clip_tensor.clone()
    frame_mean = clip_mod.mean(dim=(3, 4), keepdim=True)  # (1,3,T,1,1)
    clip_mod[:, :, frame_ids, :, :] = frame_mean[:, :, frame_ids, :, :]
    return clip_mod


def gradcam_temporal_importance(model, cam_engine, clip, class_idx):
    """
    Returns per-frame importance from Grad-CAM: (T,)
    """
    clip = clip.clone().detach().requires_grad_(True)
    cam, _, _ = cam_engine(clip, class_idx=class_idx)         # (1,T',H',W')
    cam_up = upsample_cam_to_input(cam, clip.shape)[0]        # (T,H,W)
    cam_up = cam_up.detach().cpu().numpy()
    return cam_up.mean(axis=(1, 2))


def ig_temporal_importance(model, clip, class_idx, steps=8):
    ig = integrated_gradients_video(model, clip, target_class=class_idx, steps=steps, baseline=None)
    return temporal_importance_from_ig(ig)


def deletion_curve(model, clip, ranked_frames, class_idx, num_frames):
    """
    Computes P(class_idx) after deleting top-k frames for k=0..num_frames.
    Returns probs array length num_frames+1 and AUC.
    """
    probs = []
    orig = predict_prob_of_class(model, clip, class_idx)
    probs.append(orig)

    for k in range(1, num_frames + 1):
        frames = ranked_frames[:k]
        clip_mod = occlude_frames_mean(clip, frames)
        p = predict_prob_of_class(model, clip_mod, class_idx)
        probs.append(p)

    probs = np.array(probs)
    x = np.linspace(0, 1, len(probs))
    auc = np.trapezoid(probs, x)
    return probs, auc


def faithfulness_gap_top_vs_random(model, clip, ranked_frames, class_idx, num_frames, k_list, rng=None):
    """
    Returns dict k -> (top_drop, rand_drop).
    """
    if rng is None:
        rng = np.random.RandomState(0)

    orig = predict_prob_of_class(model, clip, class_idx)
    out = {}

    for k in k_list:
        topk = ranked_frames[:k]
        clip_top = occlude_frames_mean(clip, topk)
        p_top = predict_prob_of_class(model, clip_top, class_idx)
        top_drop = orig - p_top

        randk = rng.choice(np.arange(num_frames), size=k, replace=False)
        clip_rand = occlude_frames_mean(clip, randk)
        p_rand = predict_prob_of_class(model, clip_rand, class_idx)
        rand_drop = orig - p_rand

        out[k] = (top_drop, rand_drop)

    return out


def dataset_level_auc(model, cam_engine, df_paths, device, num_frames, n_samples=20, ig_steps=8, seed=0):
    """
    Compute dataset-level mean±std AUC for Grad-CAM and IG on a subset of df_paths (expects column 'path').
    """
    rng = np.random.RandomState(seed)
    idxs = rng.permutation(len(df_paths))[:n_samples]

    auc_cam = []
    auc_ig = []

    for i in tqdm(idxs):
        path = df_paths.iloc[i]["path"]
        clip = df_paths.iloc[i].get("clip", None)
        if clip is None:
            # assume caller builds clip externally; we keep this function lightweight
            pass

    # This function is meant to be used from a notebook/script where clip loading exists.
    # Keep the heavy decoding logic in dataset.py to avoid duplication.
    raise NotImplementedError("Use this helper from a notebook/script after loading clips via dataset.py.")