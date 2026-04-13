import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    RRFRanker,
    connections,
    utility,
)

from .config import MultimodalConfig
from .utils import compute_file_hash


logger = logging.getLogger(__name__)

REQUIRED_FIELDS = [
    "id",
    "vector",
    "text",
    "image_path",
    "video_path",
    "thumbnail_base64",
    "image_base64",
    "content_hash",
    "modality",
    "metadata",
    "timestamp",
]


class AsyncMilvusMultimodalKB:
    """异步 Milvus 知识库包装。"""

    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.collection: Optional[Collection] = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="milvus")
        self._embedder = None

    async def initialize(self, embedder=None):
        self._embedder = embedder
        await self._run_sync(self._connect)
        await self._run_sync(self._ensure_collection)

    async def close(self):
        self._executor.shutdown(wait=True)

    async def _run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(func, *args, **kwargs))

    def _connect(self):
        if self.config.milvus_uri:
            db_path = Path(self.config.milvus_uri).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            lite_runtime_dir = (db_path.parent / ".milvus_lite_runtime").resolve()
            lite_runtime_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TMPDIR"] = str(lite_runtime_dir)
            os.environ["TMP"] = str(lite_runtime_dir)
            os.environ["TEMP"] = str(lite_runtime_dir)
            tempfile.tempdir = str(lite_runtime_dir)
            connections.connect(alias="default", uri=str(db_path))
            logger.info("已连接本地 Milvus Lite: %s (runtime=%s)", db_path, lite_runtime_dir)
            return

        connections.connect(alias="default", host=self.config.milvus_host, port=self.config.milvus_port)
        logger.info("已连接 Milvus: %s:%s", self.config.milvus_host, self.config.milvus_port)

    def _ensure_collection(self):
        if utility.has_collection(self.config.collection_name):
            temp_collection = Collection(self.config.collection_name)
            existing_fields = [field.name for field in temp_collection.schema.fields]
            missing_fields = set(REQUIRED_FIELDS) - set(existing_fields)

            if missing_fields:
                logger.warning("集合缺少字段 %s，删除重建...", missing_fields)
                utility.drop_collection(self.config.collection_name)
                self._create_collection()
                return

            vector_field = next((field for field in temp_collection.schema.fields if field.name == "vector"), None)
            existing_dim = vector_field.params.get("dim") if vector_field else 0
            if existing_dim != self.config.vector_dim:
                logger.warning(
                    "维度不匹配！现有:%s维，需要:%s维，删除重建...",
                    existing_dim,
                    self.config.vector_dim,
                )
                utility.drop_collection(self.config.collection_name)
                self._create_collection()
                return

            self.collection = temp_collection
            self.collection.load()
            logger.info("加载已有集合: %s", self.config.collection_name)
            return

        self._create_collection()

    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.vector_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="video_path", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="thumbnail_base64", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="image_base64", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(
            fields=fields,
            description="异步多模态知识库：支持图文视频混合检索",
            enable_dynamic_field=True,
        )

        self.collection = Collection(name=self.config.collection_name, schema=schema)

        vector_index_params = {
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "params": self.config.index_params,
        }
        if self.config.milvus_uri:
            vector_index_params = {
                "index_type": "AUTOINDEX",
                "metric_type": self.config.metric_type,
                "params": {},
            }

        self.collection.create_index(field_name="vector", index_params=vector_index_params)
        self.collection.create_index(field_name="content_hash", index_params={"index_type": "Trie"})
        self.collection.load()
        logger.info("创建集合并建立索引: %s", self.config.collection_name)

    @staticmethod
    def _fill_string_list(values: Optional[List[str]], batch_size: int) -> List[str]:
        if not values:
            return [""] * batch_size
        return [value if value is not None else "" for value in values]

    @staticmethod
    def _fill_metadata_list(values: Optional[List[Dict]], batch_size: int) -> List[Dict]:
        if not values:
            return [{} for _ in range(batch_size)]
        return [value if value is not None else {} for value in values]

    @staticmethod
    def _clean_optional_str(value: Optional[str]) -> Optional[str]:
        return value or None

    def _score_to_similarity(self, score: float) -> float:
        metric_type = (self.config.metric_type or "").upper()
        if metric_type in {"COSINE", "IP"}:
            return float(score)
        if metric_type in {"L2", "EUCLIDEAN"}:
            return 1.0 / (1.0 + float(score))
        return float(score)

    def _build_modalities(
        self,
        batch_size: int,
        texts: Optional[List[str]],
        image_paths: Optional[List[str]],
        video_paths: Optional[List[str]],
    ) -> List[str]:
        modalities = []
        for index in range(batch_size):
            has_text = texts and index < len(texts) and texts[index] is not None
            has_img = image_paths and index < len(image_paths) and image_paths[index] is not None
            has_vid = video_paths and index < len(video_paths) and video_paths[index] is not None

            if has_vid:
                modalities.append("video_text" if has_text else "video")
            elif has_text and has_img:
                modalities.append("multimodal")
            elif has_img:
                modalities.append("image")
            else:
                modalities.append("text")
        return modalities

    async def check_exists_by_hash(self, file_path: str) -> Optional[str]:
        if not file_path or not os.path.exists(file_path):
            return None

        content_hash = await self._run_sync(compute_file_hash, file_path)

        def _search_hash():
            return self.collection.query(
                expr=f'content_hash == "{content_hash}"',
                output_fields=["id", "image_path", "video_path"],
                limit=1,
            )

        existing = await self._run_sync(_search_hash)
        if existing:
            logger.info("严格去重：文件哈希匹配，已存在 ID=%s", existing[0]["id"])
            return existing[0]["id"]
        return None

    async def check_exists_by_semantic(self, vector: np.ndarray, modality: str) -> Optional[str]:
        if not self._embedder:
            logger.warning("未提供 embedder，跳过语义去重")
            return None

        def _search_similar():
            expr = None
            if modality == "image":
                expr = 'modality in ["image", "multimodal"]'
            elif modality == "video":
                expr = 'modality in ["video", "video_text"]'
            elif modality == "text":
                expr = 'modality == "text"'

            return self.collection.search(
                data=[vector.tolist()],
                anns_field="vector",
                param={"metric_type": self.config.metric_type, "params": {"ef": 64}},
                limit=1,
                expr=expr,
                output_fields=["id"],
            )

        results = await self._run_sync(_search_similar)
        if results and results[0]:
            hit = results[0][0]
            similarity = self._score_to_similarity(hit.distance)
            if similarity >= self.config.similarity_threshold:
                logger.info(
                    "语义去重：分数 %.3f，相似度 %.3f >= %s，视为重复",
                    hit.distance,
                    similarity,
                    self.config.similarity_threshold,
                )
                return hit.id
        return None

    async def insert(
        self,
        vectors: List[np.ndarray],
        texts: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        video_paths: Optional[List[str]] = None,
        thumbnails: Optional[List[str]] = None,
        image_base64_list: Optional[List[str]] = None,
        content_hashes: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
    ) -> List[str]:
        batch_size = len(vectors)
        ids = ids or [str(uuid.uuid4()) for _ in range(batch_size)]
        modalities = modalities or self._build_modalities(batch_size, texts, image_paths, video_paths)
        timestamps = [int(time.time())] * batch_size

        entities = [
            ids,
            vectors,
            self._fill_string_list(texts, batch_size),
            self._fill_string_list(image_paths, batch_size),
            self._fill_string_list(video_paths, batch_size),
            self._fill_string_list(thumbnails, batch_size),
            self._fill_string_list(image_base64_list, batch_size),
            self._fill_string_list(content_hashes, batch_size),
            modalities,
            self._fill_metadata_list(metadatas, batch_size),
            timestamps,
        ]

        def _do_insert():
            self.collection.insert(entities)
            self.collection.flush()

        await self._run_sync(_do_insert)
        logger.info("异步插入 %s 条记录", batch_size)
        return ids

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        distance_threshold: float = 0.5,
        modality_filter: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict]:
        output_fields = output_fields or [
            "id",
            "text",
            "image_path",
            "video_path",
            "modality",
            "metadata",
            "thumbnail_base64",
            "image_base64",
        ]

        expr = None
        if modality_filter == "visual":
            expr = 'modality in ["image", "video", "video_text", "multimodal"]'
        elif modality_filter == "video_all":
            expr = 'modality in ["video", "video_text"]'
        elif modality_filter:
            expr = f'modality == "{modality_filter}"'

        def _do_search():
            results = self.collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param={"metric_type": self.config.metric_type, "params": {"ef": 64}},
                limit=top_k,
                expr=expr,
                output_fields=output_fields,
            )
            hits = []
            for result in results[0]:
                similarity = self._score_to_similarity(result.distance)
                if similarity < distance_threshold:
                    continue
                hits.append({
                    "id": result.id,
                    "distance": result.distance,
                    "text": self._clean_optional_str(result.entity.get("text")),
                    "image_path": self._clean_optional_str(result.entity.get("image_path")),
                    "video_path": self._clean_optional_str(result.entity.get("video_path")),
                    "modality": self._clean_optional_str(result.entity.get("modality")),
                    "metadata": result.entity.get("metadata"),
                    "thumbnail": self._clean_optional_str(result.entity.get("thumbnail_base64")),
                    "image_base64": self._clean_optional_str(result.entity.get("image_base64")),
                })
            return hits

        return await self._run_sync(_do_search)

    async def hybrid_search(
        self,
        text_vector: Optional[np.ndarray] = None,
        image_vector: Optional[np.ndarray] = None,
        video_vector: Optional[np.ndarray] = None,
        weights: Optional[List[float]] = None,
        top_k: int = 5,
        target_modality: Optional[str] = None,
    ) -> List[Dict]:
        del weights

        reqs = []
        if text_vector is not None:
            reqs.append(AnnSearchRequest(
                data=[text_vector.tolist()],
                anns_field="vector",
                param={"metric_type": self.config.metric_type, "params": {"ef": 64}},
                limit=top_k * 3,
            ))
        if image_vector is not None:
            reqs.append(AnnSearchRequest(
                data=[image_vector.tolist()],
                anns_field="vector",
                param={"metric_type": self.config.metric_type, "params": {"ef": 64}},
                limit=top_k * 3,
            ))
        if video_vector is not None:
            reqs.append(AnnSearchRequest(
                data=[video_vector.tolist()],
                anns_field="vector",
                param={"metric_type": self.config.metric_type, "params": {"ef": 64}},
                limit=top_k * 3,
            ))
        if not reqs:
            raise ValueError("至少提供一个查询向量")

        rerank = RRFRanker(k=60)

        def _do_hybrid():
            results = self.collection.hybrid_search(
                reqs=reqs,
                rerank=rerank,
                limit=top_k * 3,
                output_fields=[
                    "id",
                    "text",
                    "image_path",
                    "video_path",
                    "modality",
                    "metadata",
                    "thumbnail_base64",
                    "image_base64",
                ],
            )

            hits = []
            for result in results[0]:
                hit = {
                    "id": result.id,
                    "text": self._clean_optional_str(result.entity.get("text")),
                    "image_path": self._clean_optional_str(result.entity.get("image_path")),
                    "video_path": self._clean_optional_str(result.entity.get("video_path")),
                    "modality": self._clean_optional_str(result.entity.get("modality")),
                    "metadata": result.entity.get("metadata"),
                    "distance": result.distance,
                    "thumbnail": self._clean_optional_str(result.entity.get("thumbnail_base64")),
                    "image_base64": self._clean_optional_str(result.entity.get("image_base64")),
                }
                if target_modality == "video" and hit["modality"] not in ["video", "video_text"]:
                    continue
                if target_modality == "image" and hit["modality"] not in ["image", "multimodal"]:
                    continue
                if target_modality == "text" and hit["modality"] != "text":
                    continue
                if target_modality == "visual" and hit["modality"] not in ["image", "video", "video_text", "multimodal"]:
                    continue
                hits.append(hit)
                if len(hits) >= top_k:
                    break
            return hits

        return await self._run_sync(_do_hybrid)

    async def delete(self, ids: List[str]):
        def _do_delete():
            self.collection.delete(f'id in {json.dumps(ids)}')

        await self._run_sync(_do_delete)
        logger.info("删除 %s 条记录", len(ids))

    async def get_stats(self) -> Dict:
        def _do_stats():
            effective_index_type = "AUTOINDEX" if self.config.milvus_uri else self.config.index_type
            return {
                "collection_name": self.config.collection_name,
                "entities_count": self.collection.num_entities,
                "vector_dim": self.config.vector_dim,
                "index_type": effective_index_type,
                "metric_type": self.config.metric_type,
            }

        return await self._run_sync(_do_stats)
