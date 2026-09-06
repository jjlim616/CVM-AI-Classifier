import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
from backend import app as api
from backend import models

client = TestClient(app)


@pytest.fixture
def installed_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "ARTIFACTS", tmp_path)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "convnext-small.pth").touch()


def test_models_show_missing_weights_without_claiming_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "ARTIFACTS", tmp_path)
    response = client.get("/api/models")
    assert response.status_code == 200
    assert len(response.json()["models"]) == 4
    assert all(not model["available"] for model in response.json()["models"])
    assert response.headers["cache-control"] == "no-store"


def test_unknown_model_and_missing_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "ARTIFACTS", tmp_path)
    response = client.post("/api/predict", data={"model_id": "../../model"}, files={"file": ("x.png", b"x", "image/png")})
    assert response.status_code == 400
    response = client.post("/api/predict", data={"model_id": "convnext-small"}, files={"file": ("x.png", b"x", "image/png")})
    assert response.status_code == 503


def test_disguised_image_rejected(installed_checkpoint):
    response = client.post("/api/predict", data={"model_id": "convnext-small"}, files={"file": ("x.png", b"not an image", "image/png")})
    assert response.status_code == 400


def test_size_rejected_before_processing():
    response = client.post("/api/predict", content=b"x", headers={"content-length": str(11 * 1024 * 1024)})
    assert response.status_code == 413


def test_cross_origin_upload_rejected():
    response = client.post("/api/predict", headers={"origin": "https://example.com"})
    assert response.status_code == 403


def test_valid_upload_reaches_inference(installed_checkpoint, monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80)).save(buffer, format="PNG")
    def fake_predict(model_id, image):
        assert model_id == "convnext-small" and image.size == (100, 80)
        return {"predicted_stage": "CS2"}
    monkeypatch.setattr(api.service, "predict", fake_predict)
    response = client.post("/api/predict", data={"model_id": "convnext-small"}, files={"file": ("x.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200
    assert response.json()["predicted_stage"] == "CS2"
