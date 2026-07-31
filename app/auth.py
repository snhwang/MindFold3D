import bcrypt
import uuid
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

from app import models
from app import schemas
from app.database import get_db

load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    # Local/replication fallback: generate an ephemeral signing key so the
    # app runs out of the box (git clone -> pip install -> uvicorn). Login
    # sessions are invalidated on every restart under this mode. For any
    # deployment, set SECRET_KEY in the environment (see env.example).
    import secrets as _secrets
    SECRET_KEY = _secrets.token_urlsafe(64)
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "SECRET_KEY is not set - generated an EPHEMERAL key for this run. "
        "Login sessions will not survive a restart. Set SECRET_KEY in your "
        "environment for any real deployment (see env.example)."
    )
elif len(SECRET_KEY) < 32:
    raise RuntimeError(
        "SECRET_KEY is too short. It must be at least 32 characters long. "
        "Use a cryptographically random value (e.g., 'openssl rand -hex 32')."
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8") if isinstance(hashed_password, str) else hashed_password,
    )


def get_password_hash(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def authenticate_user(db: Session, username: str, password: str):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    # Guest users get a lightweight object without a database lookup.
    # Each guest login receives a unique session ID embedded in the JWT so that
    # concurrent guest sessions cannot share in-memory state or DB rows.
    if payload.get("is_guest"):
        guest = models.User(username="guest", email="guest@mindfold3d", hashed_password="", is_active=True)
        guest.id = None
        guest_session_id = payload.get("guest_session_id") or str(uuid.uuid4())
        guest.session_key = f"guest-{guest_session_id}"
        return guest
    user = db.query(models.User).filter(models.User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    # Attach a stable session key for authenticated users (their integer PK as str).
    user.session_key = str(user.id)
    return user


def get_current_active_user(current_user: models.User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user 