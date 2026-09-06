from pathlib import Path
import pickle

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .inference import MAX_BYTES, InferenceService, decode_image
from .models import SPECS

torch.set_num_threads(4)
app = FastAPI(title="CVM Studio", version="2.0.0")
service = InferenceService()


@app.middleware("http")
async def local_request_policy(request: Request, call_next):
    # Same-origin browser use only. The Vite proxy preserves the browser Host.
    origin = request.headers.get("origin")
    if request.method == "POST" and origin and origin.rstrip("/") != f"{request.url.scheme}://{request.headers.get('host')}":
        return JSONResponse({"detail": "Cross-origin uploads are not allowed."}, status_code=403)
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > MAX_BYTES + 64 * 1024:
                return JSONResponse({"detail": "Choose an image smaller than 10 MB."}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid request size."}, status_code=400)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/api/health")
def health():
    return {"status": "ok", "device": "CPU"}


@app.get("/api/models")
def list_models():
    return {"models": [
        {"id": spec.id, "name": spec.name, "architecture": spec.architecture,
         "available": spec.path.is_file(), "input_size": [224, 224],
         "status": "Weights found" if spec.path.is_file() else "Weights missing"}
        for spec in SPECS.values()
    ]}


@app.post("/api/predict")
async def predict(model_id: str = Form(...), file: UploadFile = File(...)):
    try:
        if model_id not in SPECS:
            raise HTTPException(400, "Choose a supported model.")
        if not SPECS[model_id].path.is_file():
            raise HTTPException(503, "This model's weights are not installed locally.")
        raw = await file.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, "Choose an image smaller than 10 MB.")
        image = await run_in_threadpool(decode_image, raw)
        return await run_in_threadpool(service.predict, model_id, image)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except BlockingIOError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (RuntimeError, OSError, EOFError, pickle.UnpicklingError) as exc:
        raise HTTPException(503, "Model analysis failed. Verify the local checkpoint with the model validation script.") from exc
    finally:
        await file.close()


# A production build is served by the same Python process; no second server needed.
DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
