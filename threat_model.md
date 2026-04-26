# Threat Model

## Project Overview

MindFold 3D is a FastAPI application for spatial-cognition training and assessment. It serves static browser pages from `static/`, exposes JSON API endpoints from `main.py` and `auth_routes.py`, stores user accounts and long-lived stats in SQLAlchemy-backed storage, and optionally calls external services for password-reset email (`email_service.py` / Resend) and LLM-generated coaching (`llm_coaching.py`). Users can authenticate with username/password or use a guest flow.

Production assumptions for this threat model:
- The deployed service runs with `NODE_ENV=production`.
- TLS for browser-to-app traffic is provided by the platform.
- Mockup sandbox environments are not production and are out of scope unless a code path is clearly production-reachable.

## Assets

- **User accounts and session tokens** — usernames, emails, hashed passwords, password-reset tokens, and JWTs. Compromise enables impersonation and account takeover.
- **User performance data** — answer history, feature-level scorecards, response times, coaching context, and session summaries. This is user-specific behavioral data and must not leak across users.
- **Application secrets** — `SECRET_KEY`, database credentials, Resend API key, and LLM API key. Compromise would enable token forgery or abuse of third-party integrations.
- **Persistent training records** — `GameStats`, password reset rows, and any future analytics stored in the database. Integrity matters because they drive coaching and scorecards.
- **External-service trust** — outbound email and LLM requests carry application authority and sometimes user-derived content. They must not receive data from the wrong user context.

## Trust Boundaries

- **Browser ↔ API boundary** — every request body, header, query parameter, and client-side state value is untrusted. The backend must validate and scope all data server-side.
- **Public ↔ authenticated boundary** — login, registration, guest login, and password-reset routes are public; gameplay, scorecard, profile, reset, and coaching routes are authenticated. The backend must enforce this boundary without relying on the frontend.
- **Per-user boundary inside authenticated traffic** — authenticated users must be isolated from one another. In-memory runtime state, persistent stats, and generated coaching must be scoped to the acting user.
- **API ↔ database boundary** — SQLAlchemy mediates access to users, password-reset records, and game stats. The application layer must prevent cross-user reads/writes before data reaches the database.
- **API ↔ external services boundary** — `email_service.py` and `llm_coaching.py` send server-side data to third parties. Only intended data should cross this boundary, and production-safe configuration is required.
- **Production ↔ local-development boundary** — local conveniences (localhost URLs, fallback secrets, debug/test behavior, local email bypasses) must not remain effective in production.

## Scan Anchors

- **Production entry points:** `main.py`, `auth_routes.py`, `auth.py`, `database.py`, `models.py`, `static/index.html`, `static/scorecard.html`, `static/auth.html`.
- **Highest-risk areas:** auth token issuance/validation, password reset flow, in-memory `session_data` handling, per-user stats aggregation, and frontend `innerHTML` / HTML tooltip rendering paths.
- **Public surfaces:** `/register`, `/login`, `/guest-login`, `/request-password-reset`, `/reset-password`, `/`, `/home`, `/login`, `/static/*`.
- **Authenticated surfaces:** gameplay endpoints, scorecard/stat endpoints, coaching endpoints, profile, and reset endpoints in `main.py`.
- **Usually dev-only / low-priority unless production-reachable:** `.local/`, workflow files, README examples, and local-default configuration guidance. Do **not** assume `static/debug-*.html` is dev-only if it is served under `/static`.

## Threat Categories

### Spoofing

The application uses bearer JWTs for both registered and guest users. The backend must sign tokens with a strong deployment-specific secret, reject forged or stale tokens, and ensure local-development fallback behavior cannot authenticate production requests. Password-reset flows must also avoid issuing bearer-equivalent reset artifacts to the requesting client.

### Tampering

Gameplay submissions include user-controlled answer metadata, shape-feature dictionaries, and interaction telemetry. The server must validate this input and store it only in the acting user’s state. Reset endpoints and score aggregation logic must not let one authenticated user overwrite or clear another user’s live state.

### Information Disclosure

The system handles user emails, password-reset tokens, session summaries, feature-level performance, and recent attempts. API responses, scorecards, and coaching outputs must expose only the current user’s data. Local testing shortcuts, logs, or debug paths must not reveal secrets or reset tokens in production. Data sent to external LLM/email providers must be limited to the requesting user’s context.

### Denial of Service

Public authentication routes and external-service-backed coaching flows can be abused if left unbounded. The service must avoid shared mutable state that lets one user clear or poison another user’s session, and it should bound expensive or repeated requests that hit the database or external LLM APIs.

### Elevation of Privilege

Any flaw that enables JWT forgery, password-reset takeover, stored XSS in authenticated pages, or cross-user state confusion can let an attacker act with another user’s privileges. The application must enforce per-user authorization on every data path and must treat browser-rendered HTML sinks as privileged because they can access stored bearer tokens.
