from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/stat_assistant"
    OPENAI_API_KEY: str = ""
    JWT_SECRET_KEY: str = "change-me-to-a-random-secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 hours
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3001"

    model_config = {"env_file": "../../.env"}


settings = Settings()
