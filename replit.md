# MindFold 3D

A computational framework for adaptive spatial cognition assessment and training. Uses procedurally generated 3D voxel stimuli to help users train and measure 3D spatial reasoning skills (mental rotation, mirror discrimination, and configural binding).

## Architecture

- **Backend**: FastAPI (Python) serving both API and static frontend files
- **Frontend**: Vanilla JavaScript with Three.js (3D rendering) and Chart.js (analytics)
- **Database**: SQLAlchemy ORM with SQLite (default) or PostgreSQL via `DATABASE_URL`
- **Auth**: JWT-based authentication with `python-jose` and `passlib`
- **AI Coaching**: OpenAI-compatible API integration (optional, via `OPENAI_API_KEY`)

## Project Structure

- `main.py` — FastAPI app, task presentation engine, all API routes
- `auth.py` / `auth_routes.py` — JWT authentication logic
- `database.py` — SQLAlchemy engine/session setup
- `models.py` — Database ORM models
- `shape_generation.py` — Procedural 3D shape generation
- `skeleton_generation.py` — Shape skeleton generation
- `cognitive_mapping.py` — Maps shape features to cognitive difficulty
- `shape_features.py` — Shape feature definitions
- `static/` — Frontend HTML, JS, CSS, audio assets
- `metrics_framework/` — Standalone modular performance metrics library

## Running

The app runs on port 5000 via uvicorn with auto-reload:
```
python main.py
```

## Environment Variables

- `SECRET_KEY` (required) — JWT signing secret, minimum 32 chars
- `DATABASE_URL` (optional) — Defaults to `sqlite:///./mindfold.db`
- `OPENAI_API_KEY` (optional) — Enables the AI coaching feature

## Game Modes

1. **Recognition Mode** — Identify target shape among distractors
2. **Builder Mode** — Reconstruct target shape voxel-by-voxel

## Deployment

Configured for autoscale deployment with `python main.py`.
