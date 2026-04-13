import base64
import json
import logging
import os
import uuid
import asyncio
import aiohttp  # 新增：异步HTTP客户端
import cv2
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any, AsyncIterator, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
from PIL import Image
import time
from io import BytesIO
from functools import partial
import hashlib

# Milvus 客户端（保持同步，用线程池包装）
from pymilvus import (
    connections, # 连接管理器
    FieldSchema, # 字段定义
    CollectionSchema, # 集合结构
    DataType,
    Collection,
    utility,
    AnnSearchRequest,
    RRFRanker
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultimodalConfig:
    """配置类（增加异步并发参数）"""
    # vLLM/Qwen3-VL-Embedding 服务配置
    embedding_api_url: str = "http://10.8.90.116:28000/v1/embeddings"
    model_name: str = "Qwen3-VL-Embedding-8B"
    api_key: str = "EMPTY"

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "multimodal_kb"

    # 向量配置
    vector_dim: int = 4096
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    index_params: Dict = field(default_factory=lambda: {
        "M": 16,
        "efConstruction": 200
    })

    #：去重配置
    enable_deduplication: bool = True  # 是否启用地去重
    dedup_mode: str = "semantic"  # "strict"(严格哈希) 或 "semantic"(语义相似)
    similarity_threshold: float = 0.95  # 语义去重阈值（相似度>0.95视为重复）
    check_existing_before_insert: bool = True  # 插入前检查是否存在

    # 视频处理配置
    video_sample_frames: int = 8
    video_sample_strategy: str = "uniform"
    video_max_duration: int = 300

    # 新增：异步并发控制
    max_concurrent_embeds: int = 8  # 嵌入API最大并发（防止GPU OOM）
    max_concurrent_frames: int = 8  # 视频帧处理并发
    http_timeout: aiohttp.ClientTimeout = field(
        default_factory=lambda: aiohttp.ClientTimeout(total=60, connect=10)
    )


class AsyncQwen3VLEmbedder:
    """异步 Qwen3-VL-Embedding 客户端（支持高并发视频处理）"""

    def __init__(self, config: MultimodalConfig):
        self.config = config
        self._sem = asyncio.Semaphore(config.max_concurrent_embeds) #用于限制并发数量，# 1. 初始化时创建信号量（假设 max_concurrent_embeds = 4）
        self._session: Optional[aiohttp.ClientSession] = None
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}"
        }
        # 线程池用于CPU密集型操作（cv2）
        self._thread_pool = ThreadPoolExecutor(max_workers=4)

    async def initialize(self):
        """初始化 aiohttp session（必须调用）"""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=self.config.http_timeout,
                connector=aiohttp.TCPConnector(limit=20)  # 连接池限制
            )

    async def close(self):
        """清理资源"""
        if self._session:
            await self._session.close()
            self._session = None
        self._thread_pool.shutdown(wait=True)

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _compute_file_hash(self, file_path: str) -> str:
        """计算文件MD5哈希（用于严格去重）"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    ####============ CPU 密集型计算， 非 IO 密集型等待   ===========
    def _encode_image_sync(self, image_path: str) -> str:
        """同步图片编码（在线程池运行）"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _encode_frame_sync(self, frame: np.ndarray) -> str:
        """同步帧编码（在线程池运行）"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _extract_video_frames_sync(
            self,
            video_path: str,
            num_frames: int = None,
            strategy: str = None
    ) -> List[np.ndarray]:
        """同步视频帧提取（CPU密集型，在线程池运行）"""
        num_frames = num_frames or self.config.video_sample_frames
        strategy = strategy or self.config.video_sample_strategy

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        if duration > self.config.video_max_duration:
            logger.warning(f"视频时长 {duration:.1f}s 超过限制")

        frames = []
        try:
            if strategy == "uniform":
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            elif strategy == "keyframe":
                step = max(1, total_frames // (num_frames * 2))
                last_frame = None
                selected = 0
                for i in range(0, total_frames, step):
                    if selected >= num_frames:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    if last_frame is not None:
                        diff = np.mean(cv2.absdiff(frame, last_frame))
                        if diff > 15:
                            frames.append(frame)
                            selected += 1
                    else:
                        frames.append(frame)
                        selected += 1
                    last_frame = frame
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"无法从视频中提取帧: {video_path}")

        logger.info(f"提取 {len(frames)} 帧 (策略: {strategy})")
        return frames

    async def _embed_single_request(
            self,
            content: List[Dict],
            instruction: str
    ) -> np.ndarray:
        """受信号量保护的单次嵌入请求（内部方法）"""
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": content}
            ],
            "encoding_format": "float",
            "add_special_tokens": True
        }

        # 信号量控制并发，防止GPU过载，使用 async with 获取令牌
        async with self._sem:
            #关键：只有拿到令牌才能执行。受保护区域：同时最多只有max_concurrent_embeds个请求能进入这里
            try:
                async with self._session.post(#请求embedding模型
                        self.config.embedding_api_url,
                        json=payload
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    embedding = result['data'][0]['embedding']#拿到嵌入向量
                    vector = np.array(embedding, dtype=np.float32)

                    # MRL裁剪
                    if len(vector) > self.config.vector_dim:
                        vector = vector[:self.config.vector_dim]
                    elif len(vector) < self.config.vector_dim:
                        logger.warning(f"返回维度 {len(vector)} 小于配置")
                    return vector

            except Exception as e:
                logger.error(f"嵌入请求失败: {e}")
                raise

    async def embed(
            self,
            text: Optional[str] = None,
            image_path: Optional[str] = None,
            video_path: Optional[str] = None,
            video_frames: Optional[List[np.ndarray]] = None,
            instruction: str = "Represent the user's input."
    ) -> np.ndarray:
        """
        异步生成多模态嵌入（支持高并发视频帧处理）

        优化点：
        1. 视频帧提取 offload 到线程池（避免阻塞事件循环）
        2. 多帧并发嵌入（受信号量控制，默认4并发）
        3. 自动池化融合
        """
        # 视频处理分支
        if video_frames is not None or video_path is not None:
            # 步骤1：提取帧（如果是路径）→ 在线程池执行，不阻塞asyncio
            if video_frames is None:
                loop = asyncio.get_event_loop()
                video_frames = await loop.run_in_executor(
                    self._thread_pool,
                    partial(self._extract_video_frames_sync, video_path)
                )

            # 步骤2：并发嵌入所有帧（受信号量限制）
            async def embed_frame(frame: np.ndarray):
                """嵌入单帧（复用图片编码逻辑）"""
                # 帧编码（CPU操作）→ 线程池
                loop = asyncio.get_event_loop()
                base64_img = await loop.run_in_executor(
                    self._thread_pool,
                    partial(self._encode_frame_sync, frame)
                )

                content = [{
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                }]
                return await self._embed_single_request(content, instruction)

            # 限制帧处理并发数（避免瞬间爆发请求）
            frame_sem = asyncio.Semaphore(self.config.max_concurrent_frames)

            async def sem_embed_frame(frame):
                async with frame_sem:
                    return await embed_frame(frame)

            # 并发嵌入所有帧（gather自动调度）
            try:
                frame_vectors = await asyncio.gather(
                    *[sem_embed_frame(frame) for frame in video_frames],
                    return_exceptions=True  # 部分失败不全部失败
                )

                # 过滤失败的帧
                valid_vectors = [
                    v for v in frame_vectors
                    if isinstance(v, np.ndarray)
                ]
                failed_count = len(frame_vectors) - len(valid_vectors)
                if failed_count > 0:
                    logger.warning(f"{failed_count}/{len(video_frames)} 帧嵌入失败")

                if not valid_vectors:
                    raise ValueError("所有视频帧嵌入失败")

                # 时序平均池化
                video_vector = np.mean(valid_vectors, axis=0)
                if len(video_vector) > self.config.vector_dim:
                    video_vector = video_vector[:self.config.vector_dim]

                # 可选：与文本融合
                if text:
                    text_vec = await self.embed(text=text, instruction=instruction)
                    combined = 0.7 * video_vector + 0.3 * text_vec
                    return combined / np.linalg.norm(combined)

                return video_vector

            except Exception as e:
                logger.error(f"视频嵌入失败: {e}")
                raise

        # 图文处理分支
        content = []

        if image_path:
            # 图片编码 → 线程池
            loop = asyncio.get_event_loop()
            base64_img = await loop.run_in_executor(
                self._thread_pool,
                partial(self._encode_image_sync, image_path)
            )
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })

        if text:
            content.append({"type": "text", "text": text})

        if not content:
            raise ValueError("必须提供 text、image_path 或 video_path")

        return await self._embed_single_request(content, instruction)

    async def embed_batch(
            self,
            items: List[Dict[str, Any]],
            batch_size: int = 4
    ) -> List[Optional[np.ndarray]]:
        """
        异步批量嵌入（带背压控制，防止内存爆炸）

        优化：用生成器+ gather 替代逐批循环，最大化并发
        """

        async def process_single(item: Dict) -> Optional[np.ndarray]:
            try:
                if item.get("video_path"):
                    return await self.embed(
                        video_path=item.get("video_path"),
                        text=item.get("text"),
                        instruction=item.get("instruction", "Represent the user's input.")
                    )
                else:
                    return await self.embed(
                        text=item.get("text"),
                        image_path=item.get("image_path"),
                        instruction=item.get("instruction", "Represent the user's input.")
                    )
            except Exception as e:
                logger.error(f"批量处理单项失败: {e}")
                return None

        # 控制并发数，避免同时处理太多视频撑爆内存/GPU
        semaphore = asyncio.Semaphore(batch_size)

        async def sem_process(item):
            async with semaphore:
                return await process_single(item)

        # 并发处理所有项目
        results = await asyncio.gather(
            *[sem_process(item) for item in items],
            return_exceptions=True
        )

        # 转换异常为None
        final_results = []
        for r in results:
            if isinstance(r, Exception):
                final_results.append(None)
            else:
                final_results.append(r)
        return final_results

    async def embed_video_stream(
            self,
            video_path: str,
            text: Optional[str] = None,
            instruction: str = "Represent the user's input."
    ) -> np.ndarray:
        """
        流式视频嵌入：边提取帧边嵌入，减少内存占用（适合长视频）

        处理流程：提取帧 → 并发嵌入批次 → 增量池化
        """
        loop = asyncio.get_event_loop()
        frames = await loop.run_in_executor(
            self._thread_pool,
            partial(self._extract_video_frames_sync, video_path)
        )

        # 流式处理：每4帧一批，逐步融合（减少同时持有的向量数）
        batch_size = 4
        accumulated = None
        count = 0

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            # 并发嵌入本批次
            batch_vectors = await asyncio.gather(
                *[self.embed(video_frames=[f], instruction=instruction) for f in batch],
                return_exceptions=True
            )

            # 增量更新平均
            for vec in batch_vectors:
                if isinstance(vec, np.ndarray):
                    if accumulated is None:
                        accumulated = vec
                        count = 1
                    else:
                        # 增量平均公式：new_avg = (old_avg * n + new) / (n+1)
                        accumulated = (accumulated * count + vec) / (count + 1)
                        count += 1

        if accumulated is None:
            raise ValueError("无法生成视频嵌入")

        # 最终与文本融合
        if text:
            text_vec = await self.embed(text=text, instruction=instruction)
            combined = 0.7 * accumulated + 0.3 * text_vec
            return combined / np.linalg.norm(combined)

        return accumulated

##知识库CRUD操作
class AsyncMilvusMultimodalKB:
    """异步 Milvus 知识库包装（线程池 offload）"""

    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.collection: Optional[Collection] = None
        # Milvus客户端是同步的，用线程池包装
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="milvus")
        self._loop = asyncio.get_event_loop()
        self._embedder: Optional[AsyncQwen3VLEmbedder] = None  # 用于去重查询


    async def initialize(self, embedder: AsyncQwen3VLEmbedder = None):
        """异步初始化连接和集合"""
        self._embedder = embedder
        await self._run_sync(self._connect)
        await self._run_sync(self._ensure_collection)

    async def close(self):
        """清理资源"""
        self._executor.shutdown(wait=True)

    def _connect(self):
        """同步连接方法（在线程池运行）"""
        connections.connect(
            alias="default",
            host=self.config.milvus_host,
            port=self.config.milvus_port
        )
        logger.info(f"已连接 Milvus: {self.config.milvus_host}:{self.config.milvus_port}")

    def _ensure_collection(self):
        """同步集合确认（增强版：自动检测新增字段）"""
        if utility.has_collection(self.config.collection_name):
            temp_collection = Collection(self.config.collection_name)

            # 检查字段数量（新增字段会导致不匹配）
            existing_fields = [f.name for f in temp_collection.schema.fields]
            required_fields = ["id", "vector", "text", "image_path", "video_path",
                               "thumbnail_base64", "image_base64", "content_hash",  # 新增
                               "modality", "metadata", "timestamp"]

            # 检查是否包含所有必需字段
            missing_fields = set(required_fields) - set(existing_fields)

            if missing_fields:
                logger.warning(f"集合缺少字段 {missing_fields}，删除重建...")
                utility.drop_collection(self.config.collection_name)
                self._create_collection()
                return

            # 原有维度检查
            vector_field = None
            for field in temp_collection.schema.fields:
                if field.name == "vector":
                    vector_field = field
                    break

            existing_dim = vector_field.params.get("dim") if vector_field else 0

            if existing_dim != self.config.vector_dim:
                logger.warning(f"维度不匹配！现有:{existing_dim}维，需要:{self.config.vector_dim}维，删除重建...")
                utility.drop_collection(self.config.collection_name)
                self._create_collection()
            else:
                self.collection = temp_collection
                self.collection.load()
                logger.info(f"加载已有集合: {self.config.collection_name}")
        else:
            self._create_collection()

    def _create_collection(self):
        """同步创建集合（完整Schema）"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64,
                        is_primary=True, auto_id=False, nullable=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.vector_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=2048, nullable=True),
            FieldSchema(name="video_path", dtype=DataType.VARCHAR, max_length=2048, nullable=True),
            FieldSchema(name="thumbnail_base64", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
            FieldSchema(name="image_base64", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64, nullable=True),
            FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=32, default_value="unknown"),
            FieldSchema(name="metadata", dtype=DataType.JSON, nullable=True),
            FieldSchema(name="timestamp", dtype=DataType.INT64, nullable=True)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="异步多模态知识库：支持图文视频混合检索",
            enable_dynamic_field=True
        )

        self.collection = Collection(
            name=self.config.collection_name,
            schema=schema
        )

        index_params = {
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "params": self.config.index_params
        }

        self.collection.create_index(field_name="vector", index_params=index_params)

        # 为 content_hash 创建索引（加速去重查询）
        hash_index_params = {"index_type": "Trie"}
        self.collection.create_index(field_name="content_hash", index_params=hash_index_params)

        self.collection.load()
        logger.info(f"创建集合并建立索引: {self.config.collection_name}")

    async def _run_sync(self, func, *args, **kwargs):
        """辅助方法：在线程池运行同步函数"""
        return await self._loop.run_in_executor(
            self._executor,
            partial(func, *args, **kwargs)
        )

    async def check_exists_by_hash(self, file_path: str) -> Optional[str]:
        """严格去重：通过文件哈希检查是否已存在"""
        if not file_path or not os.path.exists(file_path):
            return None

        # 计算文件MD5
        def _calc_hash():
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

        content_hash = await self._run_sync(_calc_hash)

        # 查询Milvus
        def _search_hash():
            results = self.collection.query(
                expr=f'content_hash == "{content_hash}"',
                output_fields=["id", "image_path", "video_path"],
                limit=1
            )
            return results

        existing = await self._run_sync(_search_hash)
        if existing and len(existing) > 0:
            logger.info(f"严格去重：文件哈希匹配，已存在 ID={existing[0]['id']}")
            return existing[0]['id']
        return None

    async def check_exists_by_semantic(self, vector: np.ndarray, modality: str) -> Optional[str]:
        """语义去重：通过向量相似度检查是否已存在"""
        if not self._embedder:
            logger.warning("未提供 embedder，跳过语义去重")
            return None

        # 搜索相似向量
        def _search_similar():
            search_params = {
                "metric_type": self.config.metric_type,
                "params": {"ef": 64}
            }

            # 限制搜索相同模态
            expr = None
            if modality == "image":
                expr = 'modality in ["image", "multimodal"]'
            elif modality == "video":
                expr = 'modality in ["video", "video_text"]'
            elif modality == "text":
                expr = 'modality == "text"'

            results = self.collection.search(
                data=[vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=1,  # 只取最相似的
                expr=expr,
                output_fields=["id", "distance"]
            )
            return results

        results = await self._run_sync(_search_similar)

        if results and len(results[0]) > 0:
            hit = results[0][0]
            distance = hit.distance
            similarity = 1 - distance  # 余弦相似度

            if similarity >= self.config.similarity_threshold:
                logger.info(f"语义去重：相似度 {similarity:.3f} >= {self.config.similarity_threshold}，视为重复")
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
            modalities: Optional[List[str]] = None
    ) -> List[str]:
        """异步插入（offload到线程）"""
        batch_size = len(vectors)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(batch_size)]

        # 确定模态类型（包含视频判断）
        if modalities is None:
            modalities = []
            for i in range(batch_size):
                has_text = texts and i < len(texts) and texts[i] is not None
                has_img = image_paths and i < len(image_paths) and image_paths[i] is not None
                has_vid = video_paths and i < len(video_paths) and video_paths[i] is not None

                if has_vid:
                    modalities.append("video_text" if has_text else "video")
                elif has_text and has_img:
                    modalities.append("multimodal")
                elif has_img:
                    modalities.append("image")
                else:
                    modalities.append("text")

        timestamps = [int(time.time())] * batch_size

        # 构建实体（包含 content_hash）
        entities = [
            ids,
            vectors,
            texts if texts else [None] * batch_size,
            image_paths if image_paths else [None] * batch_size,
            video_paths if video_paths else [None] * batch_size,
            thumbnails if thumbnails else [None] * batch_size,
            image_base64_list if image_base64_list else [None] * batch_size,
            content_hashes if content_hashes else [None] * batch_size,  # ← 新增：第8位
            modalities,  # ← 移到第9位
            metadatas if metadatas else [None] * batch_size,  # ← 第10位
            timestamps  # ← 第11位
        ]

        def _do_insert():
            self.collection.insert(entities)
            self.collection.flush()

        await self._run_sync(_do_insert)
        logger.info(f"异步插入 {batch_size} 条记录")
        return ids

    async def search(
            self,
            query_vector: np.ndarray,
            top_k: int = 5,
            distance_threshold: float = 0.5,  # 新增：距离阈值
            modality_filter: Optional[str] = None,
            output_fields: List[str] = None
    ) -> List[Dict]:
        """异步搜索"""
        if output_fields is None:
            output_fields = ["id", "text", "image_path", "video_path", "modality", "metadata", "distance"]

        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"ef": 64, "radius": distance_threshold}
        }

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
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=output_fields
            )
            hits = []
            for result in results[0]:
                hits.append({
                    "id": result.id,
                    "distance": result.distance,
                    "text": result.entity.get("text"),
                    "image_path": result.entity.get("image_path"),
                    "video_path": result.entity.get("video_path"),
                    "modality": result.entity.get("modality"),
                    "metadata": result.entity.get("metadata"),
                    "thumbnail": result.entity.get("thumbnail_base64")
                })
            return hits

        return await self._run_sync(_do_search)

    async def hybrid_search(
            self,
            text_vector: Optional[np.ndarray] = None,
            image_vector: Optional[np.ndarray] = None,
            video_vector: Optional[np.ndarray] = None,
            weights: List[float] = None,
            top_k: int = 5,
            target_modality: Optional[str] = None
    ) -> List[Dict]:
        """异步混合搜索（修复版 - 移除 expr 参数避免版本冲突）"""
        reqs = []

        if text_vector is not None:
            reqs.append(AnnSearchRequest(
                data=[text_vector.tolist()],
                anns_field="vector",
                param={"metric_type": self.config.metric_type, "params": {"ef": 64}},
                limit=top_k * 3  # 多取一些用于客户端过滤
            ))

        if image_vector is not None:
            reqs.append(AnnSearchRequest(
                data=[image_vector.tolist()],
                anns_field="vector",
                param={"metric_type": self.config.metric_type, "params": {"ef": 64}},
                limit=top_k * 3
            ))

        if video_vector is not None:
            reqs.append(AnnSearchRequest(
                data=[video_vector.tolist()],
                anns_field="vector",
                param={"metric_type": self.config.metric_type, "params": {"ef": 64}},
                limit=top_k * 3
            ))

        if not reqs:
            raise ValueError("至少提供一个查询向量")

        rerank = RRFRanker(k=60)

        def _do_hybrid():
            # 关键修复：不再传递 expr 参数给 hybrid_search
            results = self.collection.hybrid_search(
                reqs=reqs,
                rerank=rerank,
                limit=top_k * 3,  # 多取一些用于过滤
                output_fields=["id", "text", "image_path", "video_path", "modality", "metadata", "thumbnail_base64"]
                # 注意：这里删掉了 expr=expr 参数！
            )

            # 改为在客户端根据 target_modality 过滤结果
            hits = []
            for result in results[0]:
                hit = {
                    "id": result.id,
                    "text": result.entity.get("text"),
                    "image_path": result.entity.get("image_path"),
                    "video_path": result.entity.get("video_path"),
                    "modality": result.entity.get("modality"),
                    "metadata": result.entity.get("metadata"),
                    "distance": result.distance,
                    "thumbnail": result.entity.get("thumbnail_base64")
                }

                # 客户端过滤（替代原来的 expr 服务端过滤）
                if target_modality == "video":
                    if hit["modality"] not in ["video", "video_text"]:
                        continue
                elif target_modality == "image":
                    if hit["modality"] not in ["image", "multimodal"]:
                        continue
                elif target_modality == "text":
                    if hit["modality"] != "text":
                        continue
                elif target_modality == "visual":
                    if hit["modality"] not in ["image", "video", "video_text", "multimodal"]:
                        continue

                hits.append(hit)
                if len(hits) >= top_k:
                    break

            return hits

        return await self._run_sync(_do_hybrid)


    async def delete(self, ids: List[str]):
        """异步删除"""

        def _do_delete():
            expr = f'id in {json.dumps(ids)}'
            self.collection.delete(expr)

        await self._run_sync(_do_delete)
        logger.info(f"删除 {len(ids)} 条记录")

    async def get_stats(self) -> Dict:
        """异步获取统计"""

        def _do_stats():
            return {
                "collection_name": self.config.collection_name,
                "entities_count": self.collection.num_entities,
                "vector_dim": self.config.vector_dim,
                "index_type": self.config.index_type,
                "metric_type": self.config.metric_type
            }

        return await self._run_sync(_do_stats)


class AsyncMultimodalRAGSystem:
    """完整的异步 RAG 系统"""

    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()
        self.embedder = AsyncQwen3VLEmbedder(self.config)
        self.kb = AsyncMilvusMultimodalKB(self.config)

    async def initialize(self):
        """初始化所有组件"""
        await self.embedder.initialize()
        await self.kb.initialize(embedder=self.embedder)

    async def close(self):
        """清理所有资源"""
        await self.embedder.close()
        await self.kb.close()
    ###实现异步上下文管理器逻辑__aenter__、__aexit__
    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _check_duplicate(self, text: Optional[str] = None,
                               image_path: Optional[str] = None,
                               video_path: Optional[str] = None,
                               vector: Optional[np.ndarray] = None) -> Tuple[bool, Optional[str]]:
        """
        内部方法：检查是否重复
        返回: (是否重复, 已存在的ID)
        """
        if not self.config.enable_deduplication:
            return False, None

        modality = "text"
        if video_path:
            modality = "video"
        elif image_path:
            modality = "image"

        # 严格去重模式（文件级MD5）
        if self.config.dedup_mode == "strict":
            file_path = video_path or image_path
            if file_path:
                existing_id = await self.kb.check_exists_by_hash(file_path)
                if existing_id:
                    return True, existing_id

        # 语义去重模式（向量相似度）
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
            skip_duplicate: bool = True  # 新增：是否跳过重复
    ) -> Optional[str]:

        """
        添加文档（自动去重）
        如果已存在，返回已有ID；如果不存在，插入新记录并返回新ID
        """
        # 1. 先计算向量（去重查询需要）
        vector = await self.embedder.embed(
            text=text,
            image_path=image_path,
            video_path=video_path
        )

        # 2. 检查重复
        if skip_duplicate and self.config.enable_deduplication:
            is_dup, existing_id = await self._check_duplicate(
                text=text,
                image_path=image_path,
                video_path=video_path,
                vector=vector
            )
            if is_dup:
                logger.info(f"跳过重复文档，已存在 ID: {existing_id}")
                return existing_id

        # 3. 准备内容哈希（严格去重用）
        content_hash = None
        file_path = video_path or image_path
        if file_path and os.path.exists(file_path):
            def _calc_hash():
                hash_md5 = hashlib.md5()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                return hash_md5.hexdigest()

            loop = asyncio.get_event_loop()
            content_hash = await loop.run_in_executor(None, _calc_hash)



        """异步添加文档（支持视频）"""
        if video_path:
            # 提取缩略图（异步）
            loop = asyncio.get_event_loop()
            thumbnail_b64 = None
            if extract_thumbnail:
                def _extract_thumb():
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    b64 = None
                    if ret:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(rgb)
                        buf = BytesIO()
                        pil.save(buf, format="JPEG")
                        b64 = base64.b64encode(buf.getvalue()).decode()
                    cap.release()
                    return b64

                thumbnail_b64 = await loop.run_in_executor(None, _extract_thumb)

            # 生成向量（异步嵌入）
            vector = await self.embedder.embed(video_path=video_path, text=text)

            ids = await self.kb.insert(
                vectors=[vector],
                texts=[text] if text else None,
                video_paths=[video_path],
                thumbnails=[thumbnail_b64] if thumbnail_b64 else None,
                content_hashes=[content_hash] if content_hash else None,
                metadatas=[metadata] if metadata else None
            )
            return ids[0]

        # 图文处理
        vector = await self.embedder.embed(text=text, image_path=image_path)

        img_b64 = None
        if store_image_base64 and image_path:
            def _read_img():
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')

            loop = asyncio.get_event_loop()
            img_b64 = await loop.run_in_executor(None, _read_img)  #处理CPU密集型任务

        ids = await self.kb.insert(
            vectors=[vector],
            texts=[text] if text else None,
            image_paths=[image_path] if image_path else None,
            image_base64_list=[img_b64] if img_b64 else None,
            content_hashes=[content_hash] if content_hash else None,
            metadatas=[metadata] if metadata else None
        )
        return ids[0]

    async def add_documents_batch(
            self,
            documents: List[Dict],
            max_concurrent: int = 4
    ) -> List[str]:
        """
        高性能批量添加（并发控制）

        documents格式: [{"text": "...", "image_path": "...", "video_path": "...", "metadata": {...}}, ...]
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def process_one(doc):
            async with sem:
                 return await self.add_document(
                    text=doc.get("text"),
                    image_path=doc.get("image_path"),
                    video_path=doc.get("video_path"),
                    metadata=doc.get("metadata"),
                    store_image_base64=doc.get("store_image_base64", False),
                     extract_thumbnail=doc.get("extract_thumbnail", True)
                )
        ###批量处理
        results = await asyncio.gather(
            *[process_one(doc) for doc in documents],
            return_exceptions=True
        )

        # 过滤异常
        valid_ids = [r for r in results if isinstance(r, str)]
        failed = [i for i, r in enumerate(results) if isinstance(r, Exception)]
        if failed:
            logger.error(f"批量插入失败索引: {failed}")
        return valid_ids

    async def query(
            self,
            text: Optional[str] = None,
            image_path: Optional[str] = None,
            video_path: Optional[str] = None,
            top_k: int = 5,
            distance_threshold: float = 0.5,
            filter_modality: Optional[str] = None
    ) -> List[Dict]:
        """异步跨模态检索"""
        if not text and not image_path and not video_path:
            raise ValueError("需要提供 text、image_path 或 video_path")

        query_vec = await self.embedder.embed(
            text=text,
            image_path=image_path,
            video_path=video_path
        )
        return await self.kb.search(query_vec, top_k, distance_threshold, filter_modality)

    async def video_search(
            self,
            query_video_path: str,
            target_type: str = "all",
            top_k: int = 5
    ) -> List[Dict]:
        """异步视频搜X"""
        filter_map = {
            "video": "video_all",
            "image": "image",
            "text": "text",
            "all": None
        }
        return await self.query(
            video_path=query_video_path,
            filter_modality=filter_map.get(target_type),
            top_k=top_k
        )

    async def cross_modal_search(
            self,
            text: Optional[str] = None,
            image_path: Optional[str] = None,
            video_path: Optional[str] = None,
            target_modality: Optional[str] = None,
            top_k: int = 5
    ) -> List[Dict]:
        """异步多模态融合搜索"""
        text_vec = await self.embedder.embed(text=text) if text else None
        img_vec = await self.embedder.embed(image_path=image_path) if image_path else None
        vid_vec = await self.embedder.embed(video_path=video_path) if video_path else None

        return await self.kb.hybrid_search(
            text_vector=text_vec,
            image_vector=img_vec,
            video_vector=vid_vec,
            top_k=top_k,
            target_modality=target_modality
        )

    async def get_collection_stats(self):
        return await self.kb.get_stats()


# 异步工厂函数
async def create_kb_async(
        milvus_host: str = "localhost",
        embedding_url: str = "http://10.8.90.116:28000/v1/embeddings",
        collection_name: str = "multimodal_kb",
        video_sample_frames: int = 8,
        max_concurrent_embeds: int = 4,
        enable_deduplication: bool = True,
        dedup_mode: str = "semantic"
) -> AsyncMultimodalRAGSystem:
    """异步工厂函数"""
    config = MultimodalConfig(
        milvus_host=milvus_host,
        embedding_api_url=embedding_url,
        collection_name=collection_name,
        vector_dim=4096,
        video_sample_frames=video_sample_frames,
        max_concurrent_embeds=max_concurrent_embeds,
        enable_deduplication=enable_deduplication,
        dedup_mode=dedup_mode,
        similarity_threshold=0.75 if dedup_mode == "semantic" else 1.0
    )
    rag = AsyncMultimodalRAGSystem(config)
    await rag.initialize()
    return rag


# 使用示例
async def main():
    # """演示去重功能"""
    # # 配置去重参数
    # rag = await create_kb_async(
    #     collection_name="dedup_demo",
    #     dedup_mode="semantic",  # 语义去重（相似度>0.95视为重复）
    #     # dedup_mode="strict",           # 或严格去重（MD5完全匹配）
    #     enable_deduplication=True
    # )
    #
    # try:
    #     print("=== 测试去重功能 ===")
    #
    #     # 第一次插入
    #     print("\n1. 首次插入图片...")
    #     id1 = await rag.add_document(
    #         text="一只金毛犬",
    #         image_path="./docs/gg.jpg"
    #     )
    #     print(f"   新插入 ID: {id1}")
    #
    #     # 重复插入相同图片（应该返回已有ID）
    #     print("\n2. 重复插入相同图片（应被去重）...")
    #     id2 = await rag.add_document(
    #         text="一只金毛犬",
    #         image_path="./docs/gg.jpg"  # 相同文件
    #     )
    #     print(f"   返回 ID: {id2} {'（去重成功）' if id1 == id2 else '（新插入，去重失败）'}")
    #
    #     # 插入相似图片（语义去重会拦截，严格去重会通过）
    #     print("\n3. 插入相似图片...")
    #     id3 = await rag.add_document(
    #         text="一只金毛犬",
    #         image_path="./docs/gg.jpg"  # 内容相似但文件不同
    #     )
    #     print(f"   返回 ID: {id3} {'（语义去重）' if id3 == id1 else '（视为新图片）'}")
    #
    #     # 查看统计
    #     stats = await rag.get_collection_stats()
    #     print(f"\n库中实际记录数: {stats['entities_count']}（去重后应更少）")
    #
    # finally:
    #     await rag.close()


    """完整流程：数据入库 + 多模态检索"""
    # ========== 初始化 ==========
    rag = await create_kb_async(
        collection_name="image_rag_demo",
        dedup_mode="semantic",
        enable_deduplication=True,
        max_concurrent_embeds=4
    )

    try:
        # ========== 阶段1：数据入库（插入图片到Milvus） ==========
        print("【阶段1】开始构建图片知识库...")
        # 插入文本
        print("→ 插入文本...")
        doc_text_id = await  rag.add_document(
            text="一只狗在草地上奔跑，阳光明媚",
        )
        print(f"  ✓ 插入成功，ID: {doc_text_id}")


        # 场景A：插入单张图片（带描述文本）
        print("→ 插入单张图片（带描述）...")
        doc_id1 = await rag.add_document(
            text="一只狗在草地上奔跑，阳光明媚",  # 文本描述（可选但推荐）
            image_path="docs/gg.jpg",
            # metadata={
            #     "source": "camera",
            #     "category": "animal",
            #     "tags": ["dog", "golden_retriever", "outdoor"],
            #     "created_at": "2024-01-15"
            # },
            store_image_base64=False  # 存储base64用于前端预览（可选）
        )
        print(f"  ✓ 插入成功，ID: {doc_id1}")

        # 场景B：批量插入多张图片（高效并发）
        print("\n→ 批量插入图片库...")
        batch_docs = [
            {
                "image_path": "./docs/cat_01.jpg",
                "text": "一只橘猫在沙发上睡觉",
                "metadata": {"category": "animal", "tags": ["cat", "indoor"]}
            },
            {
                "image_path": "./docs/car_01.jpg",
                "text": "红色跑车在城市街道上行驶",
                "metadata": {"category": "vehicle", "tags": ["car", "sports_car", "red"]}
            },
            {
                "image_path": "./docs/food_01.jpg",
                "text": "精致的日式寿司摆盘",
                "metadata": {"category": "food", "tags": ["sushi", "japanese", "restaurant"]}
            },
            {
                "image_path": "./docs/beach_01.jpg",
                "text": "夕阳下的海滩，金色的沙滩和蓝色的海水",
                "metadata": {"category": "landscape", "tags": ["beach", "sunset", "ocean"]}
            }
        ]

        # 并发插入，max_concurrent控制并发数（防止GPU过载）
        inserted_ids = await rag.add_documents_batch(batch_docs, max_concurrent=4)
        print(f"  ✓ 批量插入完成，共 {len(inserted_ids)} 张图片")

        # 场景C：插入纯文本（作为对比基准，可选）
        print("\n→ 插入纯文本描述...")
        text_id = await rag.add_document(
            text="这不是图片，只是一段关于自然风光的描述文字",
            metadata={"type": "text_only", "category": "description"}
        )
        print(f"  ✓ 文本插入成功，ID: {text_id}")

        # 查看库统计信息
        stats = await rag.get_collection_stats()
        print(f"\n【库状态】当前共有 {stats['entities_count']} 条记录")

        # ========== 阶段2：多模态检索 ==========
        print("\n" + "="*50)
        print("【阶段2】开始多模态检索测试...")

        # 检索1：文搜图（Text-to-Image）
        print("\n【检索1】文搜图：查询 '一只狗在草地上'...")
        text_results = await rag.query(
            text="一只猫在沙发上睡觉",
            distance_threshold=0.4,
            filter_modality="image",  # 只返回图片
            top_k=3
        )
        print(f"找到 {len(text_results)} 个结果：")
        for i, hit in enumerate(text_results, 1):
            similarity = 1 - hit['distance']  # 余弦相似度转换
            print(f"  {i}. ID: {hit['id'][:8]}... | 相似度: {similarity:.3f}")
            print(f"     路径: {hit.get('image_path', 'N/A')}")
            if hit.get('text'):
                print(f"     描述: {hit['text'][:30]}...")

        # 检索2：图搜图（Image-to-Image）
        print("\n【检索2】图搜图：用查询图片搜索相似图片...")
        # 假设有一张查询图片
        query_img = "./docs/beach_01.jpg"
        if os.path.exists(query_img):
            image_results = await rag.query(
                image_path=query_img,
                # filter_modality="image",
                top_k=3
            )
            print(f"找到 {len(image_results)} 个相似图片：")
            for hit in image_results:
                print(f"  - {hit.get('image_path')} (相似度: {1-hit['distance']:.3f})")
        else:
            print("  （查询图片不存在，跳过）")

        # 检索3：混合检索（文本 + 参考图片）
        print("\n【检索3】混合检索：文本描述 + 参考图片...")
        # 准备查询：既提供文字又提供图片
        hybrid_results = await rag.cross_modal_search(
            text="精致的日式寿司摆盘",           # 文本查询
            image_path="./docs/food_01.jpg",  # 参考图片（可选）
            target_modality="image",                  # 目标返回图片
            top_k=3
        )
        print(f"混合检索找到 {len(hybrid_results)} 个结果：")
        for hit in hybrid_results:
            print(f"  - ID: {hit['id'][:8]} | 模态: {hit['modality']} | 相似度: {1-hit['distance']:.3f}")

        # 检索4：文搜所有（图片+文本）
        print("\n【检索4】语义搜索：查询 '某个文本'（返回所有类型）...")
        all_results = await rag.query(
            text="一只狗在草地上奔跑，阳光明媚",
            filter_modality=None,  # 不限制模态，返回所有
            top_k=5,

        )
        print(f"找到 {len(all_results)} 条相关记录：")
        for hit in all_results:
            content_type = "图片" if hit['modality'] in ['image', 'multimodal'] else "文本"
            print(f"  - [{content_type}] {hit.get('text', 'N/A')[:40]}")

    finally:
        # 确保关闭资源
        await rag.close()
        print("\n【完成】资源已释放")


if __name__ == "__main__":
    asyncio.run(main())