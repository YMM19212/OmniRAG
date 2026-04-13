import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Union

import aiohttp
import cv2
import numpy as np

from .config import MultimodalConfig
from .utils import encode_file_base64, encode_frame_base64


logger = logging.getLogger(__name__)


class AsyncQwen3VLEmbedder:
    """异步 embedding 客户端，当前默认接 Jina Embeddings v4。"""

    def __init__(self, config: MultimodalConfig):
        self.config = config
        self._sem = asyncio.Semaphore(config.max_concurrent_embeds)
        self._session: Optional[aiohttp.ClientSession] = None
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }
        self._thread_pool = ThreadPoolExecutor(max_workers=4)

    async def initialize(self):
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=self.config.http_timeout,
                connector=aiohttp.TCPConnector(limit=20),
            )

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None
        self._thread_pool.shutdown(wait=True)

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _run_in_thread(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, partial(func, *args))

    def _extract_video_frames_sync(
        self,
        video_path: str,
        num_frames: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> List[np.ndarray]:
        num_frames = num_frames or self.config.video_sample_frames
        strategy = strategy or self.config.video_sample_strategy

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        if duration > self.config.video_max_duration:
            logger.warning("视频时长 %.1fs 超过限制", duration)

        frames: List[np.ndarray] = []
        try:
            if strategy == "uniform":
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            elif strategy == "keyframe":
                step = max(1, total_frames // (num_frames * 2))
                last_frame = None
                selected = 0
                for index in range(0, total_frames, step):
                    if selected >= num_frames:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    if last_frame is None or np.mean(cv2.absdiff(frame, last_frame)) > 15:
                        frames.append(frame)
                        selected += 1
                    last_frame = frame
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"无法从视频中提取帧: {video_path}")

        logger.info("提取 %s 帧 (策略: %s)", len(frames), strategy)
        return frames

    async def _embed_single_request(
        self,
        input_item: Union[str, Dict[str, str]],
        task: Optional[str] = None,
    ) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("Embedder 尚未初始化，请先调用 initialize()。")

        payload = {
            "model": self.config.model_name,
            "input": [input_item],
            "embedding_type": "float",
        }
        if task:
            payload["task"] = task
        # 旧内网 Qwen3-VL 调用方法先保留为参考，当前不再启用。
        # payload = {
        #     "model": self.config.model_name,
        #     "messages": [
        #         {"role": "system", "content": [{"type": "text", "text": instruction}]},
        #         {"role": "user", "content": content},
        #     ],
        #     "encoding_format": "float",
        #     "add_special_tokens": True,
        # }

        async with self._sem:
            async with self._session.post(self.config.embedding_api_url, json=payload) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error("嵌入请求失败: status=%s body=%s", response.status, error_text[:1200])
                    response.raise_for_status()
                result = await response.json()

        embedding = result["data"][0]["embedding"]
        vector = np.array(embedding, dtype=np.float32)
        if len(vector) > self.config.vector_dim:
            return vector[:self.config.vector_dim]
        if len(vector) < self.config.vector_dim:
            logger.warning("返回维度 %s 小于配置", len(vector))
        return vector

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    async def _embed_text(self, text: str) -> np.ndarray:
        return await self._embed_single_request({"text": text}, task=self.config.embedding_task)

    async def _embed_image_data_uri(self, data_uri: str) -> np.ndarray:
        return await self._embed_single_request({"image": data_uri})

    async def embed(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        video_frames: Optional[List[np.ndarray]] = None,
        instruction: str = "Represent the user's input.",
    ) -> np.ndarray:
        if video_frames is not None or video_path is not None:
            if video_frames is None and video_path is not None:
                video_frames = await self._run_in_thread(self._extract_video_frames_sync, video_path)

            async def embed_frame(frame: np.ndarray):
                base64_img = await self._run_in_thread(encode_frame_base64, frame)
                return await self._embed_image_data_uri(f"data:image/jpeg;base64,{base64_img}")

            frame_sem = asyncio.Semaphore(self.config.max_concurrent_frames)

            async def sem_embed_frame(frame: np.ndarray):
                async with frame_sem:
                    return await embed_frame(frame)

            try:
                frame_vectors = await asyncio.gather(
                    *[sem_embed_frame(frame) for frame in video_frames or []],
                    return_exceptions=True,
                )
            except Exception as exc:
                logger.error("视频嵌入失败: %s", exc)
                raise

            valid_vectors = [item for item in frame_vectors if isinstance(item, np.ndarray)]
            failed_count = len(frame_vectors) - len(valid_vectors)
            if failed_count > 0:
                logger.warning("%s/%s 帧嵌入失败", failed_count, len(video_frames or []))
            if not valid_vectors:
                raise ValueError("所有视频帧嵌入失败")

            video_vector = self._normalize_vector(np.mean(valid_vectors, axis=0))

            if text:
                text_vec = await self._embed_text(text)
                combined = 0.7 * video_vector + 0.3 * self._normalize_vector(text_vec)
                return self._normalize_vector(combined)

            return video_vector

        vectors: List[np.ndarray] = []
        if image_path:
            base64_img = await self._run_in_thread(encode_file_base64, image_path)
            image_vec = await self._embed_image_data_uri(f"data:image/jpeg;base64,{base64_img}")
            vectors.append(self._normalize_vector(image_vec))
        if text:
            text_vec = await self._embed_text(text)
            vectors.append(self._normalize_vector(text_vec))
        if not vectors:
            raise ValueError("必须提供 text、image_path 或 video_path")
        if len(vectors) == 1:
            return vectors[0]

        # Jina Embeddings API 会分别处理文本和图像，这里做简单加权融合。
        if text and image_path and len(vectors) == 2:
            combined = 0.4 * vectors[0] + 0.6 * vectors[1]
            return self._normalize_vector(combined)
        return self._normalize_vector(np.mean(vectors, axis=0))

    async def embed_batch(
        self,
        items: List[Dict[str, Any]],
        batch_size: int = 4,
    ) -> List[Optional[np.ndarray]]:
        async def process_single(item: Dict[str, Any]) -> Optional[np.ndarray]:
            try:
                if item.get("video_path"):
                    return await self.embed(
                        video_path=item.get("video_path"),
                        text=item.get("text"),
                        instruction=item.get("instruction", "Represent the user's input."),
                    )
                return await self.embed(
                    text=item.get("text"),
                    image_path=item.get("image_path"),
                    instruction=item.get("instruction", "Represent the user's input."),
                )
            except Exception as exc:
                logger.error("批量处理单项失败: %s", exc)
                return None

        semaphore = asyncio.Semaphore(batch_size)

        async def sem_process(item: Dict[str, Any]):
            async with semaphore:
                return await process_single(item)

        results = await asyncio.gather(*[sem_process(item) for item in items], return_exceptions=True)
        return [None if isinstance(item, Exception) else item for item in results]

    async def embed_video_stream(
        self,
        video_path: str,
        text: Optional[str] = None,
        instruction: str = "Represent the user's input.",
    ) -> np.ndarray:
        frames = await self._run_in_thread(self._extract_video_frames_sync, video_path)
        batch_size = 4
        accumulated = None
        count = 0

        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size]
            batch_vectors = await asyncio.gather(
                *[self.embed(video_frames=[frame], instruction=instruction) for frame in batch],
                return_exceptions=True,
            )
            for vector in batch_vectors:
                if isinstance(vector, np.ndarray):
                    if accumulated is None:
                        accumulated = vector
                        count = 1
                    else:
                        accumulated = (accumulated * count + vector) / (count + 1)
                        count += 1

        if accumulated is None:
            raise ValueError("无法生成视频嵌入")

        if text:
            text_vec = await self._embed_text(text)
            combined = 0.7 * self._normalize_vector(accumulated) + 0.3 * self._normalize_vector(text_vec)
            return self._normalize_vector(combined)

        return self._normalize_vector(accumulated)
