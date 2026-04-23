# MindFold 3D

## Project Overview

MindFold 3D is an adaptive spatial cognition assessment and training platform. It uses procedurally generated 3D voxel stimuli and a layered cognitive architecture to test and train various aspects of human spatial reasoning, including mental rotation, mirror discrimination, and working memory.

## Architecture

### Backend
- **Framework:** FastAPI (Python)
- **Server:** Uvicorn on port 5000
- **Database:** SQLite (local dev) / PostgreSQL (production)
- **ORM:** SQLAlchemy
- **Auth:** JWT tokens via python-jose, password hashing via bcrypt/passlib
- **Email:** Resend (password reset)
- **AI Coaching:** OpenAI-compatible LLM integration

### Frontend
- **3D Rendering:** Three.js for interactive 3D voxel shapes
- **Language:** Vanilla JavaScript (ES6+)
- **Served from:** `static/` directory via FastAPI StaticFiles

## Key Files
- `main.py` — FastAPI app entry point, all API endpoints, shape generation orchestration
- `shape_generation.py` — Procedural 3D shape generation logic
- `skeleton_generation.py` — Shape skeleton generation (skeleton-first method)
- `cognitive_mapping.py` — Difficulty specs and cognitive profile mapping
- `shape_features.py` — Shape feature analysis
- `auth.py` / `auth_routes.py` — Authentication logic and routes
- `database.py` — Database engine and session management
- `models.py` — SQLAlchemy database models
- `schemas.py` — Pydantic schemas
- `llm_coaching.py` — LLM-based cognitive coaching
- `email_service.py` — Email service integration
- `metrics_framework/` — Standalone performance metrics library

## Workflows
- **Start application:** `python main.py` on port 5000

## Game Modes
- **Recognition Mode:** Users identify which of multiple presented shapes matches a target shape (under rotations, viewpoints, or mirror distractors)
- **Builder Mode:** Users reconstruct a 3D target shape voxel-by-voxel

## Dependencies
All Python dependencies are listed in `requirements.txt`. Install via `pip install -r requirements.txt`.
