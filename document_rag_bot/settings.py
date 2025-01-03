import enum
from pathlib import Path
from tempfile import gettempdir
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from document_rag_bot.services.chat_models.settings import ChatModelSettings
from document_rag_bot.services.embedding_models.settings import EmbeddingModelSettings

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO

    # This variable is used to define
    # multiproc_dir. It's required for [uvi|guni]corn projects.
    prometheus_dir: Path = TEMP_DIR / "prom"

    # Sentry's configuration.
    sentry_dsn: Optional[str] = None
    sentry_sample_rate: float = 1.0

    # Variables for API
    api_host: str = "document_rag_bot-api"
    api_port: int = 8000
    api_workers_count: int = 1
    api_worker_timeout: int = 36000
    api_reload: bool = False
    api_db_readonly: bool = False

    # Embedding models
    embedding_model: EmbeddingModelSettings

    # Chat models
    chat_model: ChatModelSettings

    # Reranker
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    # Qdrant settings
    qdrant_scheme: str = "http"
    qdrant_host: str = "document_rag_bot-qdrant"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = True
    qdrant_api_key: Optional[str] = None
    qdrant_upload_batch_size: int = 20
    qdrant_upload_max_retries: int = 3
    qdrant_upload_parallel: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DOCUMENT_RAG_BOT_",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_parse_none_str="null",
        extra="allow",
    )


settings = Settings()  # type: ignore
