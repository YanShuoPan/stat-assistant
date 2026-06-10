import sys
import logging

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/stat_assistant"
    OPENAI_API_KEY: str = ""
    JWT_SECRET_KEY: str = "change-me-to-a-random-secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 hours
    DIFY_API_KEY: str = ""
    DIFY_BASE_URL: str = "https://api.dify.ai/v1"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3001"

    model_config = {"env_file": "../../.env"}


settings = Settings()

_INSECURE_DEFAULTS = {"change-me-to-a-random-secret", ""}

if settings.JWT_SECRET_KEY in _INSECURE_DEFAULTS:
    logger.critical(
        "JWT_SECRET_KEY is not set or uses the insecure default. "
        "Set a strong random secret in your .env file. Exiting."
    )
    sys.exit(1)
