from datetime import timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

import auth
import models
import schemas
from database import engine, get_db

# Create tables if they don't exist (guest users and game stats)
models.Base.metadata.create_all(bind=engine)

router = APIRouter(tags=["Session"])


@router.post("/start-session", response_model=schemas.Token)
def start_session(db: Session = Depends(get_db)):
    """Create a new anonymous guest user and issue a session token.

    This is the only authentication endpoint exposed in the public build
    of MindFold 3D. Each call creates an independent guest user whose
    game stats persist for the life of the issued token.
    """
    user = auth.create_guest_user(db)
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
