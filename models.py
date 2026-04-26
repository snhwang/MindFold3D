from sqlalchemy import Boolean, Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func

from database import Base


class User(Base):
    """Guest user for the public build of MindFold 3D.

    The public build does not support registration or credentials. Each
    game session creates a new anonymous user whose stats are keyed by a
    random UUID-derived username.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String, default="")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GameStats(Base):
    __tablename__ = "game_stats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    total_questions = Column(Integer, default=0)
    correct_answers = Column(Integer, default=0)
    feature_stats = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
