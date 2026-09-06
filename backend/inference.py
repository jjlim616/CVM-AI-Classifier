import base64
import io
import threading
import time
import warnings

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .models import CLASSES, SPECS, load_model, preprocessing

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


def decode_image(raw: bytes) -> Image.Image:
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Choose an image smaller than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise ValueError("Only PNG and JPEG images are supported.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Choose an image with at most 20 million pixels.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ValueError("The file could not be read as a PNG or JPEG image.") from exc


def png_data(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def gradcam(model, tensor):
    activations = []
    handle = model.target_layer.register_forward_hook(lambda _m, _i, output: activations.append(output))
    try:
        with torch.enable_grad():
            logits = model(tensor.requires_grad_(True))
            if logits.shape != (1, 6) or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid model output")
            index = int(logits.argmax(dim=1).item())
            features = activations[0]
            gradients = torch.autograd.grad(logits[0, index], features)[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * features).sum(dim=1, keepdim=True).relu()
            cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
            if not np.isfinite(cam).all():
                raise RuntimeError("Invalid Grad-CAM output")
            cam -= cam.min()
            peak = float(cam.max())
            if peak > 1e-12:
                cam /= peak
            else:
                cam.fill(0)
            return logits.detach().softmax(dim=1)[0].tolist(), cam, peak > 1e-12
    finally:
        handle.remove()


def colorize(cam):
    # Blue → cyan → yellow → red, matching the legacy jet visualization.
    red = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return Image.fromarray(np.uint8(np.stack([red, green, blue], axis=-1) * 255))


class InferenceService:
    """One cached model, one inference at a time; avoids unbounded RAM use."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.model_id = None

    def predict(self, model_id: str, image: Image.Image):
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("An analysis is already running. Try again shortly.")
        try:
            start = time.perf_counter()
            spec = SPECS[model_id]
            if self.model_id != model_id:
                self.model = None
                self.model_id = None
                self.model = load_model(spec)
                self.model_id = model_id
            tensor = preprocessing(spec)(image).unsqueeze(0)
            scores, cam, has_signal = gradcam(self.model, tensor)
            # Display the same grayscale geometry as the tensor to align overlays.
            source = image.resize((224, 224), Image.Resampling.BILINEAR).convert("L").convert("RGB") if spec.resize_first else image.convert("L").resize((224, 224), Image.Resampling.BILINEAR).convert("RGB")
            return {
                "model_id": model_id,
                "model_name": spec.name,
                "predicted_stage": CLASSES[int(np.argmax(scores))],
                "scores": [{"stage": stage, "score": score} for stage, score in zip(CLASSES, scores)],
                "original": png_data(source),
                "heatmap": png_data(colorize(cam)),
                "heatmap_has_signal": has_signal,
                "elapsed_ms": round((time.perf_counter() - start) * 1000),
                "input_size": [224, 224],
                "device": "CPU",
            }
        finally:
            self.lock.release()
