from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # Data‐lake roots
    MARKET_PATH: str = Field("data_lake/market", env="MARKET_PATH")
    ONCHAIN_PATH: str = Field("data_lake/onchain", env="ONCHAIN_PATH")
    SOCIAL_PATH: str = Field("data_lake/social", env="SOCIAL_PATH")
    NEWS_PATH: str = Field("data_lake/news", env="NEWS_PATH")

    # FastAPI
    HOST: str = Field("0.0.0.0", env="INGEST_HOST")
    PORT: int = Field(8000, env="INGEST_PORT")

    # CORS
    CORS_ORIGINS: list[str] = Field(["*"], env="CORS_ORIGINS")

    # Prometheus
    METRICS_PATH: str = Field("/metrics", env="METRICS_PATH")
    # Logging
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
settings = Settings()