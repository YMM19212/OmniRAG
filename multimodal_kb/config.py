import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import aiohttp


@dataclass
class MultimodalConfig:
    """项目配置。"""

    embedding_api_url: str = "https://api.jina.ai/v1/embeddings"
    model_name: str = "jina-embeddings-v4"
    api_key: str = field(default_factory=lambda: os.getenv("JINA_API_KEY", ""))
    embedding_task: str = "text-matching"

    # 默认使用本地 Milvus Lite；如需切换远程 Milvus，可清空 milvus_uri 再使用 host/port。
    milvus_uri: str = "./data/multimodal_kb.db"
    milvus_host: Optional[str] = "localhost"
    milvus_port: Optional[int] = 19530
    collection_name: str = "multimodal_kb"

    vector_dim: int = 2048
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    index_params: Dict[str, Any] = field(default_factory=lambda: {
        "M": 16,
        "efConstruction": 200,
    })

    enable_deduplication: bool = True
    dedup_mode: str = "semantic"
    similarity_threshold: float = 0.95
    check_existing_before_insert: bool = True

    video_sample_frames: int = 8
    video_sample_strategy: str = "uniform"
    video_max_duration: int = 300

    max_concurrent_embeds: int = 8
    max_concurrent_frames: int = 8
    http_timeout: aiohttp.ClientTimeout = field(
        default_factory=lambda: aiohttp.ClientTimeout(total=60, connect=10)
    )
