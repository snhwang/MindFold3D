from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid
from fastapi.security import OAuth2PasswordRequestForm

from app import schemas
from app import models
from app import auth
from app import email_service
from app.database import get_db, engine

# Create tables if they don't exist
models.Base.metadata.create_all(bind=engine)

router = APIRouter(tags=["Authentication"])


@router.post("/register", response_model=schemas.UserResponse)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check if username or email already exists
    db_user_by_username = db.query(models.User).filter(models.User.username == user.username).first()
    db_user_by_email = db.query(models.User).filter(models.User.email == user.email).first()
    
    if db_user_by_username:
        raise HTTPException(status_code=400, detail="Username already registered")
    if db_user_by_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@router.post("/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/guest-login", response_model=schemas.Token)
def guest_login():
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    guest_session_id = str(uuid.uuid4())
    access_token = auth.create_access_token(
        data={"sub": "guest", "is_guest": True, "guest_session_id": guest_session_id},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/logout")
def logout(response: Response, current_user: models.User = Depends(auth.get_current_active_user)):
    # We could blacklist the token here if we track tokens in the database
    # For simplicity, we'll just clear the cookie
    response.delete_cookie(key="access_token")
    return {"message": "Successfully logged out"}


@router.post("/request-password-reset")
def request_password_reset(request: schemas.PasswordResetRequest, db: Session = Depends(get_db)):
    if not email_service.email_enabled():
        import logging
        logging.getLogger(__name__).warning(
            "Password reset requested but email delivery is not configured. "
            "No reset token was created. Configure RESEND_API_KEY to enable password resets."
        )
        return {"message": "If the email exists, a password reset link has been sent."}

    user = db.query(models.User).filter(models.User.email == request.email).first()
    if not user:
        # Still return success to prevent email enumeration
        return {"message": "If the email exists, a password reset link has been sent."}
    
    # Create password reset token
    reset_token = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(hours=24)
    
    # Save reset token in database
    db_reset = models.PasswordReset(
        user_id=user.id,
        reset_code=reset_token,
        expires_at=expires_at
    )
    db.add(db_reset)
    db.commit()

    email_service.send_password_reset_email(user.email, reset_token)
    return {"message": "If the email exists, a password reset link has been sent."}


@router.post("/reset-password")
def reset_password(reset_data: schemas.PasswordReset, db: Session = Depends(get_db)):
    # Find the reset token
    reset_request = db.query(models.PasswordReset).filter(
        models.PasswordReset.reset_code == reset_data.token,
        models.PasswordReset.is_used == False,
        models.PasswordReset.expires_at > datetime.utcnow()
    ).first()
    
    if not reset_request:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    
    # Update the user's password
    user = db.query(models.User).filter(models.User.id == reset_request.user_id).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    
    user.hashed_password = auth.get_password_hash(reset_data.new_password)
    
    # Mark the reset token as used
    reset_request.is_used = True
    
    db.commit()
    return {"message": "Password has been reset successfully"} 