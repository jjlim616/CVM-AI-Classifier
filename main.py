"""Start the local CVM Studio API and its built Vue interface."""
from pathlib import Path
import uvicorn

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    if not (root / "frontend" / "dist" / "index.html").exists():
        print("Frontend build missing. Run: npm --prefix frontend ci && npm --prefix frontend run build")
        print("The API will still be available; Vite development can run separately.")
    print("CVM Studio: http://127.0.0.1:8000")
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, access_log=False)
