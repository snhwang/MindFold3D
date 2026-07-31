"""MindFold3D launcher.

Keeps `python main.py` working from the repo root (local runs, Replit).
The application lives in app/main.py; core generation science in mindfold3d/.
"""
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=3001)
