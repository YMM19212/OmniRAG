import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import MultimodalConfig
from .embedder import AsyncQwen3VLEmbedder
from .store import AsyncMilvusMultimodalKB
from .utils import compute_file_hash, encode_file_base64, extract_video_thumbnail_base64


logger = logging.getLogger(__name__)


class AsyncMultimodalRAGSystem:
    """组合 embedding 与向量库能力的高层服务。"""

    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()
        self.embedder = AsyncQwen3VLEmbedder(self.config)
        self.kb = AsyncMilvusMultimodalKB(self.config)

    async def initialize(self):
        await self.embedder.initialize()
        try:
            await self.kb.initialize(embedder=self.embedder)
        except Exception:
            await self.embedder.close()
            raise

    async def close(self):
        await self.embedder.close()
        await self.kb.close()

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _run_in_thread(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)

    async def _check_duplicate(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        vector: Optional[np.ndarray] = None,
    ) -> Tuple[bool, Optional[str]]:
        if not self.config.enable_deduplication:
            return False, None

        modality = "text"
        if video_path:
            modality = "video"
        elif image_path:
            modality = "image"

        if self.config.dedup_mode == "strict":
            file_path = video_path or image_path
            if file_path:
                existing_id = await self.kb.check_exists_by_hash(file_path)
                if existing_id:
                    return True, existing_id
        elif self.config.dedup_mode == "semantic" and vector is not None:
            existing_id = await self.kb.check_exists_by_semantic(vector, modality)
            if existing_id:
                return True, existing_id

        return False, None

    async def add_document(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
        store_image_base64: bool = False,
        extract_thumbnail: bool = True,
        skip_duplicate: bool = True,
    ) -> Optional[str]:
        vector = await self.embedder.embed(text=text, image_path=image_path, video_path=video_path)

        if skip_duplicate and self.config.enable_deduplication:
            is_dup, existing_id = await self._check_duplicate(
                text=text,
                image_path=image_path,
                video_path=video_path,
                vector=vector,
            )
            if is_dup:
                logger.info("跳过重复文档，已存在 ID: %s", existing_id)
                return existing_id

        content_hash = None
        file_path = video_path or image_path
        if file_path and os.path.exists(file_path):
            content_hash = await self._run_in_thread(compute_file_hash, file_path)

        if video_path:
            thumbnail_b64 = None
            if extract_thumbnail:
                thumbnail_b64 = await self._run_in_thread(extract_video_thumbnail_base64, video_path)
            ids = await self.kb.insert(
                vectors=[vector],
                texts=[text] if text else None,
                video_paths=[video_path],
                thumbnails=[thumbnail_b64] if thumbnail_b64 else None,
                content_hashes=[content_hash] if content_hash else None,
                metadatas=[metadata] if metadata else None,
            )
            return ids[0]

        img_b64 = None
        if store_image_base64 and image_path:
            img_b64 = await self._run_in_thread(encode_file_base64, image_path)

        ids = await self.kb.insert(
            vectors=[vector],
            texts=[text] if text else None,
            image_paths=[image_path] if image_path else None,
            image_base64_list=[img_b64] if img_b64 else None,
            content_hashes=[content_hash] if content_hash else None,
            metadatas=[metadata] if metadata else None,
        )
        return ids[0]

    async def add_documents_batch(self, documents: List[Dict], max_concurrent: int = 4) -> List[str]:
        sem = asyncio.Semaphore(max_concurrent)

        async def process_one(doc: Dict):
            async with sem:
                return await self.add_document(
                    text=doc.get("text"),
                    image_path=doc.get("image_path"),
                    video_path=doc.get("video_path"),
                    metadata=doc.get("metadata"),
                    store_image_base64=doc.get("store_image_base64", False),
                    extract_thumbnail=doc.get("extract_thumbnail", True),
                    skip_duplicate=doc.get("skip_duplicate", True),
                )

        results = await asyncio.gather(*[process_one(doc) for doc in documents], return_exceptions=True)
        valid_ids = [item for item in results if isinstance(item, str)]
        failed_indexes = [index for index, item in enumerate(results) if isinstance(item, Exception)]
        if failed_indexes:
            logger.error("批量插入失败索引: %s", failed_indexes)
        return valid_ids

    async def query(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        top_k: int = 5,
        distance_threshold: float = 0.5,
        filter_modality: Optional[str] = None,
    ) -> List[Dict]:
        if not text and not image_path and not video_path:
            raise ValueError("需要提供 text、image_path 或 video_path")

        query_vec = await self.embedder.embed(text=text, image_path=image_path, video_path=video_path)
        return await self.kb.search(query_vec, top_k, distance_threshold, filter_modality)

    async def video_search(self, query_video_path: str, target_type: str = "all", top_k: int = 5) -> List[Dict]:
        filter_map = {
            "video": "video_all",
            "image": "image",
            "text": "text",
            "all": None,
        }
        return await self.query(video_path=query_video_path, filter_modality=filter_map.get(target_type), top_k=top_k)

    async def cross_modal_search(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        target_modality: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        text_vec = await self.embedder.embed(text=text) if text else None
        img_vec = await self.embedder.embed(image_path=image_path) if image_path else None
        vid_vec = await self.embedder.embed(video_path=video_path) if video_path else None
        return await self.kb.hybrid_search(
            text_vector=text_vec,
            image_vector=img_vec,
            video_vector=vid_vec,
            top_k=top_k,
            target_modality=target_modality,
        )

    async def get_collection_stats(self):
        return await self.kb.get_stats()


async def create_kb_async(
    milvus_uri: Optional[str] = None,
    milvus_host: Optional[str] = None,
    embedding_url: Optional[str] = None,
    collection_name: Optional[str] = None,
    video_sample_frames: Optional[int] = None,
    max_concurrent_embeds: Optional[int] = None,
    enable_deduplication: Optional[bool] = None,
    dedup_mode: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> AsyncMultimodalRAGSystem:
    default_config = MultimodalConfig()
    config = MultimodalConfig(
        milvus_uri=default_config.milvus_uri if milvus_uri is None else (milvus_uri or None),
        milvus_host=default_config.milvus_host if milvus_host is None else (milvus_host or None),
        embedding_api_url=embedding_url or default_config.embedding_api_url,
        model_name=model_name or default_config.model_name,
        api_key=default_config.api_key if api_key is None else api_key,
        collection_name=collection_name or default_config.collection_name,
        vector_dim=default_config.vector_dim,
        video_sample_frames=video_sample_frames or default_config.video_sample_frames,
        max_concurrent_embeds=max_concurrent_embeds or default_config.max_concurrent_embeds,
        enable_deduplication=(
            default_config.enable_deduplication if enable_deduplication is None else enable_deduplication
        ),
        dedup_mode=dedup_mode or default_config.dedup_mode,
        similarity_threshold=(
            default_config.similarity_threshold
            if (dedup_mode or default_config.dedup_mode) == "semantic"
            else 1.0
        ),
    )
    rag = AsyncMultimodalRAGSystem(config)
    await rag.initialize()
    return rag
