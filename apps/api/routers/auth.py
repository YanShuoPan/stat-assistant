from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from auth import create_access_token, decode_token, get_current_user, hash_password, verify_password
from database import get_db
from models import User
from schemas import LoginRequest, Token, UserCreate, UserResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])

VALID_ROLES = {"admin", "researcher", "viewer"}


def _optional_current_user(
    authorization: str | None = Header(None),
    db: Session = Depends(get_db),
) -> User | None:
    """Try to extract user from Bearer token, return None if absent/invalid."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    user_id = decode_token(authorization[7:])
    if user_id is None:
        return None
    return db.query(User).filter(User.id == user_id).first()


@router.post("/register", response_model=UserResponse, status_code=201)
def register(
    body: UserCreate,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(_optional_current_user),
):
    """Register a new user.

    - If no users exist yet, the first account becomes admin automatically.
    - Otherwise, only admins can create new accounts.
    """
    if body.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}")

    user_count = db.query(User).count()

    if user_count > 0:
        if current_user is None or current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can register new users",
            )

    existing = db.query(User).filter(User.username == body.username).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")

    # First user is always admin
    role = "admin" if user_count == 0 else body.role

    user = User(
        username=body.username,
        hashed_password=hash_password(body.password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == body.username).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token({"sub": str(user.id)})
    return Token(access_token=token)


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    return current_user
