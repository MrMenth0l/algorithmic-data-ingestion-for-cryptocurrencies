from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    twitter_api_key: str | None = Field(None, env="TWITTER_API_KEY")
    twitter_api_secret: str | None = Field(None, env="TWITTER_API_SECRET")
    twitter_bearer: str | None = Field(None, env="TWITTER_BEARER")
    reddit_client_id: str | None = Field(None, env="REDDIT_CLIENT_ID")
    reddit_client_secret: str | None = Field(None, env="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str | None = Field(None, env="REDDIT_USER_AGENT")
    data_lake_path: str = Field("data_lake", env="DATA_LAKE_PATH")
    news_api_key: str | None = Field(None, env="NEWS_API_KEY")
    glassnode_api_key: str | None = Field(None, env="GLASSNODE_API_KEY")
    covalent_api_key: str | None = Field(None, env="COVALENT_API_KEY")

    # Data lake paths
    MARKET_PATH: str = Field("data_lake/market", env="MARKET_PATH")
    ONCHAIN_PATH: str = Field("data_lake/onchain", env="ONCHAIN_PATH")
    SOCIAL_PATH: str = Field("data_lake/social", env="SOCIAL_PATH")
    NEWS_PATH: str = Field("data_lake/news", env="NEWS_PATH")

    # Redis Feature Store settings
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_password: str | None = Field(None, env="REDIS_PASSWORD")

    # Feature Store TTL (in seconds)
    feature_ttl_seconds: int = Field(60, env="FEATURE_TTL_SECONDS")

    market_path: str = Field("data_lake/market", env="MARKET_PATH")
    onchain_path: str = Field("data_lake/onchain", env="ONCHAIN_PATH")
    social_path: str = Field("data_lake/social", env="SOCIAL_PATH")
    news_path: str = Field("data_lake/news", env="NEWS_PATH")

    #fastAPI
    ingest_host: str = Field("0.0.0.0", env="INGEST_HOST")
    ingest_port: int = Field(8000, env="INGEST_PORT")
    #CORS
    cors_origins: list[str] = Field(["*"], env="CORS_ORIGINS")
    #Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    #Metrics/Prometheus
    metrics_path: str = Field("/metrics", env="METRICS_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
settings = Settings()