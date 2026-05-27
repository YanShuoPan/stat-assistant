from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)

# DO Dev Database (PG16): connected user lacks CREATE on public schema.
# Fix: grant ourselves CREATE by altering schema ownership if we're db owner,
# or fall back to granting CREATE privilege.
if not settings.DATABASE_URL.startswith("sqlite"):
    try:
        with engine.connect() as conn:
            # Get current user and try to grant CREATE on public schema
            user = conn.execute(text("SELECT current_user")).scalar()
            conn.execute(text(f'ALTER SCHEMA public OWNER TO "{user}"'))
            conn.commit()
    except Exception:
        try:
            with engine.connect() as conn:
                user = conn.execute(text("SELECT current_user")).scalar()
                conn.execute(text(f'GRANT ALL ON SCHEMA public TO "{user}"'))
                conn.commit()
        except Exception:
            pass  # Will fail at table creation with a clear error

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
