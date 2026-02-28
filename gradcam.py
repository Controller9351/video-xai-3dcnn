import torch
import torch.nn.functional as F


class GradCAM3D:
    """
    3D Grad-CAM for torchvision video models like r3d_18.
    Produces CAM in feature-map resolution, which you can upsample to (T,H,W).
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fwd = target_layer.register_forward_hook(forward_hook)
        self.bwd = target_layer.register_full_backward_hook(backward_hook)

    def remove(self):
        self.fwd.remove()
        self.bwd.remove()

    def __call__(self, x, class_idx=None):
        """
        x: (B,3,T,H,W)
        returns cam: (B,T',H',W') normalized to [0,1]
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward()

        A = self.activations        # (B,C,T',H',W')
        G = self.gradients          # (B,C,T',H',W')
        w = G.mean(dim=(2, 3, 4), keepdim=True)

        cam = (w * A).sum(dim=1)    # (B,T',H',W')
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, logits.detach(), class_idx


def upsample_cam_to_input(cam, input_shape):
    """
    cam: (B,T',H',W')
    input_shape: tuple (B,3,T,H,W)
    returns cam_up: (B,T,H,W)
    """
    B, C, T, H, W = input_shape
    cam_up = F.interpolate(
        cam.unsqueeze(1),
        size=(T, H, W),
        mode="trilinear",
        align_corners=False
    ).squeeze(1)
    return cam_up