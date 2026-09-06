"""Architectures recovered from the original FYP notebooks.

No pretrained downloads: every parameter must match a local checkpoint strictly.
"""
from dataclasses import dataclass
from pathlib import Path
import os

import timm
import torch
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("CVM_ARTIFACTS_DIR", ROOT / "artifacts"))
CLASSES = [f"CS{i}" for i in range(1, 7)]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    architecture: str
    filename: str
    notebook: str
    resize_first: bool = False

    @property
    def path(self):
        return ARTIFACTS / "checkpoints" / self.filename


SPECS = {
    s.id: s for s in (
        ModelSpec("convnext-small", "ConvNeXt Small", "ConvNeXt", "convnext-small.pth", "convnext-small.ipynb"),
        ModelSpec("densenet121", "DenseNet121", "DenseNet", "densenet121.pth", "densenet121-v4.ipynb"),
        ModelSpec("mobilenet-v2", "MobileNetV2", "MobileNet", "mobilenet-v2.pth", "mobilenetv2-1.ipynb"),
        ModelSpec("efficientnet-b1", "EfficientNet-B1", "EfficientNet", "efficientnet-b1.pth", "jj-cvm-efficientnetb1-v2.ipynb", True),
    )
}


class CVMModel(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        if model_id == "convnext-small":
            self.convnext = timm.create_model("convnext_small", pretrained=False)
            self.convnext.head.fc = nn.Sequential(
                nn.Dropout(0.4), nn.Linear(self.convnext.head.fc.in_features, 512),
                nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.4), nn.Linear(512, 6),
            )
            self.backbone_name = "convnext"
        elif model_id == "densenet121":
            self.densenet = models.densenet121(weights=None)
            n = self.densenet.classifier.in_features
            self.densenet.classifier = nn.Sequential(
                nn.BatchNorm1d(n), nn.Dropout(0.4), nn.Linear(n, 512),
                nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 6),
            )
            self.backbone_name = "densenet"
        elif model_id == "mobilenet-v2":
            self.mobilenet = models.mobilenet_v2(weights=None)
            self.mobilenet.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.mobilenet.last_channel, 6))
            self.backbone_name = "mobilenet"
        elif model_id == "efficientnet-b1":
            self.efficientnet = timm.create_model("efficientnet_b1", pretrained=False)
            n = self.efficientnet.classifier.in_features
            self.efficientnet.classifier = nn.Sequential(
                nn.Linear(n, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 6),
            )
            self.backbone_name = "efficientnet"
        else:
            raise ValueError("Unsupported model")

    def forward(self, image):
        return getattr(self, self.backbone_name)(image)

    @property
    def target_layer(self):
        if self.backbone_name == "convnext":
            return self.convnext.stages[-1].blocks[-1]
        if self.backbone_name == "densenet":
            # Hook the final convolution, before DenseNet's in-place output ReLU.
            return self.densenet.features.denseblock4.denselayer16.conv2
        if self.backbone_name == "mobilenet":
            return self.mobilenet.features[-1]
        return self.efficientnet.conv_head


def preprocessing(spec: ModelSpec):
    resize = transforms.Resize((224, 224), antialias=True)
    grayscale = transforms.Grayscale(num_output_channels=3)
    # EfficientNet's notebook resizes before grayscale; preserve that difference.
    return transforms.Compose([
        *([resize, grayscale] if spec.resize_first else [grayscale, resize]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model(spec: ModelSpec):
    model = CVMModel(spec.id)
    state = torch.load(spec.path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    # Input gradients keep CAM functional even when all model weights are frozen.
    model.requires_grad_(False)
    return model
