from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
_pool_kwargs = (
    {"pool_size": 3, "max_overflow": 2, "pool_pre_ping": True, "pool_recycle": 1800}
    if not settings.DATABASE_URL.startswith("sqlite")
    else {}
)
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args, **_pool_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
