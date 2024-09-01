import os
from functools import lru_cache
from typing import Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class ABProfile(BaseModel):
    chat_model: str
    embed_model: str
    bm25_weight: float = 0.4
    embedding_weight: float = 0.6
    top_k: int = 6


class Settings(BaseModel):
    # GitHub Models (default path)
    use_github_models: bool = os.getenv("USE_GITHUB_MODELS", "true").lower() == "true"
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    github_models_base_url: str = os.getenv("GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference")
    github_chat_model: str = os.getenv("GITHUB_CHAT_MODEL", "openai/gpt-4.1")
    github_embed_model: str = os.getenv("GITHUB_EMBED_MODEL", "text-embedding-3-large")

    # OpenAI (direct)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("CHAT_MODEL", "openai/gpt-4.1")

    # Azure OpenAI (optional fallback)
    use_azure_openai: bool = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    azure_chat_deployment: str = os.getenv("AZURE_CHAT_DEPLOYMENT", "")
    azure_embed_deployment: str = os.getenv("AZURE_EMBED_DEPLOYMENT", "")
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "6"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.45"))
    embedding_weight: float = float(os.getenv("EMBEDDING_WEIGHT", "0.55"))
    log_dir: str = os.getenv("LOG_DIR", "logs")
    data_dir: str = os.getenv("DATA_DIR", "data")
    ab_profiles: Dict[str, ABProfile] = Field(
        default_factory=lambda: {
            "control": ABProfile(
                chat_model="openai/gpt-4.1",
                embed_model="text-embedding-3-small",
                bm25_weight=0.45,
                embedding_weight=0.55,
                top_k=6,
            ),
            "fast": ABProfile(
                chat_model="openai/gpt-4.1",
                embed_model="text-embedding-3-small",
                bm25_weight=0.5,
                embedding_weight=0.5,
                top_k=5,
            ),
            "quality": ABProfile(
                chat_model="openai/gpt-4.1",
                embed_model="text-embedding-3-small",
                bm25_weight=0.4,
                embedding_weight=0.6,
                top_k=8,
            ),
        }
    )
    pricing: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
            "gpt-4o": {"prompt": 2.5, "completion": 5.0},
            "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
            "text-embedding-3-small": {"prompt": 0.02, "completion": 0.0},
            "text-embedding-3-large": {"prompt": 0.13, "completion": 0.0},
        }
    )
    # Serper (Google Search wrapper)
    serper_api_key: str = os.getenv("SERPER_API_KEY", "")
    # URL fetch / scraping defaults
    allowed_url_regexes: list[str] = Field(
        default_factory=lambda: [
            r"^https://([a-zA-Z0-9-]+\.)*(wikipedia\.org|arxiv\.org|docs\.python\.org|developer\.mozilla\.org|readthedocs\.io|httpbin\.org|jsonplaceholder\.typicode\.com|example\.com)(/.*)?$"
        ]
    )
    robots_policy: str = os.getenv("ROBOTS_POLICY", "respect")  # respect | ignore
    robots_user_agent: str = os.getenv("ROBOTS_USER_AGENT", "GenericFetchBot/1.0")
    max_fetch_time_seconds: float = float(os.getenv("MAX_FETCH_TIME_SECONDS", "8.0"))
    max_content_size_bytes: int = int(os.getenv("MAX_CONTENT_SIZE_BYTES", str(2 * 1024 * 1024)))  # 2MB
    rate_limit_requests_per_minute: int = int(os.getenv("RATE_LIMIT_RPM", "20"))
    rate_limit_burst: int = int(os.getenv("RATE_LIMIT_BURST", "10"))
    per_domain_rate_limits: Dict[str, Dict[str, int]] = Field(
        default_factory=lambda: {
            "wikipedia.org": {"requests_per_minute": 60, "burst": 20},
            "arxiv.org": {"requests_per_minute": 30, "burst": 10},
            "docs.python.org": {"requests_per_minute": 30, "burst": 10},
            "developer.mozilla.org": {"requests_per_minute": 30, "burst": 10},
            "httpbin.org": {"requests_per_minute": 20, "burst": 10},
        }
    )
    scrape_cleaning_mode: str = os.getenv("SCRAPE_CLEANING_MODE", "readability")  # readability | basic_bs4
    ingest_fetched_to_rag: bool = os.getenv("INGEST_FETCHED_TO_RAG", "false").lower() == "true"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
