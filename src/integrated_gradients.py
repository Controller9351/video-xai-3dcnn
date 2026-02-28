import torch


def integrated_gradients_video(model, x, target_class, steps=8, baseline=None):
    """
    Integrated Gradients for video input.

    x: (1,3,T,H,W) normalized tensor
    baseline: same shape or None -> zeros
    steps: number of integration steps

    returns:
      ig: (1,3,T,H,W)
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)

    alphas = torch.linspace(0, 1, steps + 1, device=x.device).view(-1, 1, 1, 1, 1, 1)
    x_scaled = baseline + alphas * (x - baseline)      # (steps+1,1,3,T,H,W)
    x_scaled = x_scaled.squeeze(1)                     # (steps+1,3,T,H,W)
    x_scaled.requires_grad_(True)

    logits = model(x_scaled)                           # (steps+1,num_classes)
    score = logits[:, target_class].sum()

    grads = torch.autograd.grad(score, x_scaled)[0]    # (steps+1,3,T,H,W)

    avg_grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = avg_grads.mean(dim=0, keepdim=True)    # (1,3,T,H,W)

    ig = (x - baseline) * avg_grads
    return ig


def temporal_importance_from_ig(ig):
    """
    ig: (1,3,T,H,W)
    returns importance per frame: (T,)
    """
    return ig.abs().mean(dim=(1, 3, 4)).detach().cpu().numpy()[0]