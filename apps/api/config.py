from pydantic_settings import BaseSettings


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
