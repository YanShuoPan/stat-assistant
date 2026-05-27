from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)

# PG16+ revokes CREATE on public schema by default; grant it for DO Dev Database
if not settings.DATABASE_URL.startswith("sqlite"):
    try:
        with engine.connect() as conn:
            conn.execute(text("GRANT CREATE ON SCHEMA public TO CURRENT_USER"))
            conn.commit()
    except Exception:
        pass  # already granted or superuser

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
