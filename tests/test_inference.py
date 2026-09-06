import io
import threading

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from backend.inference import InferenceService, decode_image, gradcam


class ToyModel(nn.Module):
    """A single known feature; class 0 depends only on its positive pixels."""
    def __init__(self, zero=False):
        super().__init__()
        self.features = nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            self.features.weight.fill_(0 if zero else 1)

    @property
    def target_layer(self):
        return self.features

    def forward(self, tensor):
        score = self.features(tensor).mean((2, 3))
        return torch.cat([score, score * 0, score * 0, score * 0, score * 0, score * 0], dim=1)


def test_gradcam_localizes_known_feature_and_removes_hooks():
    model = ToyModel().eval()
    tensor = torch.zeros(1, 1, 224, 224)
    tensor[:, :, 80:140, 60:120] = 1
    scores, cam, signal = gradcam(model, tensor)
    assert signal
    assert cam[100, 90] == pytest.approx(1)
    assert cam[10, 10] == 0
    assert sum(scores) == pytest.approx(1)
    assert scores[0] > scores[1]
    assert not model.features._forward_hooks


def test_zero_cam_is_finite_and_reports_no_signal():
    _, cam, signal = gradcam(ToyModel(zero=True), torch.ones(1, 1, 224, 224))
    assert not signal
    assert np.isfinite(cam).all() and not cam.any()


def test_hook_removed_when_forward_raises():
    model = ToyModel()
    def fail(_):
        raise RuntimeError("failed")
    model.forward = fail
    with pytest.raises(RuntimeError):
        gradcam(model, torch.ones(1, 1, 224, 224))
    assert not model.features._forward_hooks


@pytest.mark.parametrize("raw", [b"", b"not an image", b"<svg></svg>"])
def test_invalid_images_rejected(raw):
    with pytest.raises(ValueError):
        decode_image(raw)


def test_real_format_checked_and_rgba_accepted():
    buffer = io.BytesIO()
    Image.new("RGBA", (10, 20)).save(buffer, format="PNG")
    image = decode_image(buffer.getvalue())
    assert image.mode == "RGB" and image.size == (10, 20)
    buffer = io.BytesIO()
    Image.new("RGB", (10, 20)).save(buffer, format="GIF")
    with pytest.raises(ValueError, match="Only PNG"):
        decode_image(buffer.getvalue())


def test_service_rejects_concurrent_requests():
    service = InferenceService()
    with service.lock:
        with pytest.raises(BlockingIOError):
            service.predict("convnext-small", Image.new("RGB", (224, 224)))
