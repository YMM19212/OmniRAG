import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or default


def _get_env_optional_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or None


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value.strip())


def _get_env_optional_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return None
    return int(stripped)


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value.strip())


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value}")


def _get_embedding_api_key() -> str:
    return _get_env_optional_str("OMNIRAG_EMBEDDING_API_KEY") or os.getenv("JINA_API_KEY", "")


def _get_index_params() -> Dict[str, Any]:
    return {
        "M": _get_env_int("OMNIRAG_INDEX_M", 16),
        "efConstruction": _get_env_int("OMNIRAG_INDEX_EF_CONSTRUCTION", 200),
    }


@dataclass
class MultimodalConfig:
    """项目配置。"""

    embedding_api_url: str = field(
        default_factory=lambda: _get_env_str("OMNIRAG_EMBEDDING_API_URL", "https://api.jina.ai/v1/embeddings")
    )
    model_name: str = field(
        default_factory=lambda: _get_env_str("OMNIRAG_EMBEDDING_MODEL_NAME", "jina-embeddings-v4")
    )
    api_key: str = field(default_factory=_get_embedding_api_key)
    embedding_task: str = field(default_factory=lambda: _get_env_str("OMNIRAG_EMBEDDING_TASK", "text-matching"))

    # 默认使用本地 Milvus Lite；如需切换远程 Milvus，可清空 milvus_uri 再使用 host/port。
    milvus_uri: Optional[str] = field(
        default_factory=lambda: _get_env_optional_str("OMNIRAG_MILVUS_URI", "./data/multimodal_kb.db")
    )
    milvus_host: Optional[str] = field(
        default_factory=lambda: _get_env_optional_str("OMNIRAG_MILVUS_HOST", "localhost")
    )
    milvus_port: Optional[int] = field(
        default_factory=lambda: _get_env_optional_int("OMNIRAG_MILVUS_PORT", 19530)
    )
    collection_name: str = field(default_factory=lambda: _get_env_str("OMNIRAG_COLLECTION_NAME", "multimodal_kb"))

    vector_dim: int = field(default_factory=lambda: _get_env_int("OMNIRAG_VECTOR_DIM", 2048))
    index_type: str = field(default_factory=lambda: _get_env_str("OMNIRAG_INDEX_TYPE", "HNSW"))
    metric_type: str = field(default_factory=lambda: _get_env_str("OMNIRAG_METRIC_TYPE", "COSINE"))
    index_params: Dict[str, Any] = field(default_factory=_get_index_params)

    enable_deduplication: bool = field(
        default_factory=lambda: _get_env_bool("OMNIRAG_ENABLE_DEDUPLICATION", True)
    )
    dedup_mode: str = field(default_factory=lambda: _get_env_str("OMNIRAG_DEDUP_MODE", "semantic"))
    similarity_threshold: float = field(
        default_factory=lambda: _get_env_float("OMNIRAG_SIMILARITY_THRESHOLD", 0.95)
    )
    check_existing_before_insert: bool = field(
        default_factory=lambda: _get_env_bool("OMNIRAG_CHECK_EXISTING_BEFORE_INSERT", True)
    )

    video_sample_frames: int = field(default_factory=lambda: _get_env_int("OMNIRAG_VIDEO_SAMPLE_FRAMES", 8))
    video_sample_strategy: str = field(
        default_factory=lambda: _get_env_str("OMNIRAG_VIDEO_SAMPLE_STRATEGY", "uniform")
    )
    video_max_duration: int = field(default_factory=lambda: _get_env_int("OMNIRAG_VIDEO_MAX_DURATION", 300))

    max_concurrent_embeds: int = field(default_factory=lambda: _get_env_int("OMNIRAG_MAX_CONCURRENT_EMBEDS", 8))
    max_concurrent_frames: int = field(default_factory=lambda: _get_env_int("OMNIRAG_MAX_CONCURRENT_FRAMES", 8))
    http_timeout: aiohttp.ClientTimeout = field(
        default_factory=lambda: aiohttp.ClientTimeout(
            total=_get_env_int("OMNIRAG_HTTP_TIMEOUT_TOTAL", 60),
            connect=_get_env_int("OMNIRAG_HTTP_TIMEOUT_CONNECT", 10),
        )
    )
