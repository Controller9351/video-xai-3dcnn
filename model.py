import torch
import torch.nn as nn
import torchvision


def build_r3d18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Returns an R3D-18 model for video classification.
    Input expected: (B, 3, T, H, W)
    """
    # Prefer new torchvision weights API, fallback if unavailable
    if pretrained:
        try:
            model = torchvision.models.video.r3d_18(weights="KINETICS400_V1")
        except Exception:
            model = torchvision.models.video.r3d_18(pretrained=True)
    else:
        model = torchvision.models.video.r3d_18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_fc(model: nn.Module):
    for p in model.fc.parameters():
        p.requires_grad = True


def unfreeze_layers(model: nn.Module, layer_names=("layer3", "layer4", "fc")):
    """
    Unfreeze parameters whose name contains any of layer_names strings.
    Example: ("layer3","layer4","fc")
    """
    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in layer_names)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)