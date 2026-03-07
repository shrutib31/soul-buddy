"""
ORM models for the Auth DB (souloxy-web schema).

Only the tables relevant to Soul Buddy's conversation context are modelled here.
All models use AuthBase so SQLAlchemy keeps them bound to the auth DB engine,
separate from the Data DB models that use Base.

Relevant tables
---------------
  users                    — basic identity (id, supabase_uid, full_name, role)
  user_personality_profiles — evaluated personality JSONB, keyed by supabase_uid
  user_detailed_profiles   — demographics, health history, hobbies; keyed by user_id (FK → users.id)
"""

from sqlalchemy import Integer, String, Date, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, TEXT
from sqlalchemy.orm import Mapped, mapped_column

from .auth_base import AuthBase


class AuthUser(AuthBase):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    supabase_uid: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str | None] = mapped_column(String(50), nullable=True)


class UserPersonalityProfile(AuthBase):
    __tablename__ = "user_personality_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    supabase_uid: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    personality_profile_data: Mapped[dict] = mapped_column(JSONB, nullable=False)


class UserDetailedProfile(AuthBase):
    __tablename__ = "user_detailed_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, unique=True)

    # Demographics
    first_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    date_of_birth: Mapped[Date | None] = mapped_column(Date, nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    age_group: Mapped[str | None] = mapped_column(String(20), nullable=True)
    gender: Mapped[str | None] = mapped_column(String(50), nullable=True)
    pronouns: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Location & Language
    country: Mapped[str | None] = mapped_column(String(100), nullable=True)
    timezone: Mapped[str | None] = mapped_column(String(100), nullable=True)
    languages: Mapped[list | None] = mapped_column(ARRAY(TEXT), nullable=True)
    communication_language: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Background
    education_level: Mapped[str | None] = mapped_column(String(50), nullable=True)
    occupation: Mapped[str | None] = mapped_column(String(100), nullable=True)
    marital_status: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Interests
    hobbies: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    interests: Mapped[list | None] = mapped_column(ARRAY(TEXT), nullable=True)

    # Health history (JSONB — flexible structure)
    mental_health_history: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    physical_health_history: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
