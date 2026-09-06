"""Run local checkpoint/CAM smoke checks. Never exports the supplied image."""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from backend.inference import InferenceService, decode_image
from backend.models import SPECS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Authorized local PNG or JPEG")
    args = parser.parse_args()
    image = decode_image(args.image.read_bytes())
    torch.set_num_threads(4)
    service = InferenceService()
    failed = False
    for model_id, spec in SPECS.items():
        try:
            result = service.predict(model_id, image)
            scores = [entry["score"] for entry in result["scores"]]
            assert len(scores) == 6 and all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
            assert math.isclose(sum(scores), 1, abs_tol=1e-5)
            assert result["heatmap"].startswith("data:image/png;base64,")
            print(f"PASS {spec.name}: {result['predicted_stage']}, CAM signal={result['heatmap_has_signal']}, {result['elapsed_ms']} ms")
        except Exception as exc:
            failed = True
            print(f"FAIL {spec.name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
