"""MindFold3D launcher.

Keeps `python main.py` working from the repo root (local runs, Replit).
The application lives in app/main.py; core generation science in mindfold3d/.

`app` is re-exported at module level so ASGI runners that import this
module directly (`uvicorn main:app`, Replit deployments, gunicorn) find
it. Without the re-export those runners fail with:
    Attribute "app" not found in module "main"
"""
from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
