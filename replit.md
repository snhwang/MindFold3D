# MindFold 3D

## Project Overview

MindFold 3D is an adaptive spatial cognition assessment and training platform. It uses procedurally generated 3D voxel stimuli and a layered cognitive architecture to test and train various aspects of human spatial reasoning, including mental rotation, mirror discrimination, and working memory.

This is the **public (guest-only) build** of MindFold 3D, the codebase accompanying the paper on the skeleton-first shape generation engine. It is a demo / research distribution: there is no user registration or password flow — each visitor gets an anonymous session.

## Architecture

### Backend
- **Framework:** FastAPI (Python)
- **Server:** Uvicorn on port 5000
- **Database:** SQLite (local dev) / PostgreSQL (Replit / production)
- **ORM:** SQLAlchemy
- **Auth:** anonymous guest sessions only. Each POST `/start-session` creates a new DB-backed guest user and returns a JWT signed with `SECRET_KEY`. No registration, no password, no email.
- **AI Coaching (optional):** OpenAI-compatible LLM integration (OpenAI, Groq, Together, local Ollama, etc.) via `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` env vars. The in-app brain icon is disabled if no LLM is configured.

### Frontend
- **3D Rendering:** Three.js for interactive 3D voxel shapes
- **Language:** Vanilla JavaScript (ES6+)
- **Served from:** `static/` directory via FastAPI StaticFiles
- **Splash page:** `static/auth.html` (served at `/login`) with a single **Start Game** button

## Key Files
- `main.py` — FastAPI app entry point, all API endpoints, shape generation orchestration
- `shape_generation.py` — Procedural 3D shape generation, feature analysis, canonical-form comparison (rotation-invariant dedup), voxel-nudge fallback
- `skeleton_generation.py` — Skeleton-first shape generation (the paper's active method)
- `cognitive_mapping.py` — Difficulty specs, skeleton spec construction, distractor perturbation
- `shape_features.py` — Pydantic model for measured shape features
- `auth.py` / `auth_routes.py` — Guest-session helpers and the `/start-session` route
- `database.py` — Database engine and session management
- `models.py` — SQLAlchemy models (User for guest rows, GameStats for per-session stats)
- `schemas.py` — Pydantic Token schema
- `llm_coaching.py` — Optional LLM-based cognitive coaching layer
- `metrics_framework/` — Standalone performance metrics library used by the Performance Scorecard
- `benchmark_fidelity.py`, `fidelity_study.py` — Fidelity assessment scripts for the skeleton-first method

## Workflows
- **Start application:** `python main.py` on port 5000 (wired up in `.replit`)

## Game Modes
- **Recognition Mode:** 4AFC — users identify which of multiple presented shapes matches a target (under rotations, viewpoints, or mirror distractors)
- **Builder Mode:** Users reconstruct a 3D target shape voxel-by-voxel
- **Performance Scorecard:** Per-feature success rates across the session to identify cognitive strengths and weak areas

## Dependencies
All Python dependencies are listed in `requirements.txt`. Install via `pip install -r requirements.txt`.

## Environment Variables
See `env.example` for the full list. The only required one is `SECRET_KEY` (JWT signing). Everything else has sensible defaults or is optional (`DATABASE_URL`, `LLM_*`).

## License
Licensed under Open Core Ventures Source Available License (OCVSAL) v1.0. See `LICENSE`. Production use requires a commercial agreement with the copyright holder. Commercial licensing inquiries: ottinfo@psu.edu (Penn State Office of Technology Transfer).
