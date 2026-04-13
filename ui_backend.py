import asyncio
import hashlib
import threading
from dataclasses import asdict
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, Iterable, List, Optional

from multimodal_kb import AsyncMultimodalRAGSystem, MultimodalConfig


class StreamlitKBClient:
    """在后台事件循环中持有异步知识库对象，向 Streamlit 暴露同步接口。"""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._rag: Optional[AsyncMultimodalRAGSystem] = None
        self._config: Optional[MultimodalConfig] = None
        self._status: Dict[str, Any] = {
            "state": "disconnected",
            "message": "Knowledge base is not initialized.",
            "last_error": None,
        }

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _initialize(self, config: MultimodalConfig):
        previous_rag = self._rag
        previous_config = self._config
        new_rag = AsyncMultimodalRAGSystem(config)

        try:
            await new_rag.initialize()
        except Exception:
            try:
                await new_rag.close()
            except Exception:
                pass
            self._rag = previous_rag
            self._config = previous_config
            raise

        if previous_rag is not None:
            await previous_rag.close()

        self._rag = new_rag
        self._config = config

    def initialize(self, config: MultimodalConfig):
        self._status = {
            "state": "connecting",
            "message": "Initializing knowledge base connection...",
            "last_error": None,
        }
        try:
            self._run(self._initialize(config))
        except Exception as exc:
            self._status = {
                "state": "error",
                "message": "Knowledge base initialization failed.",
                "last_error": str(exc),
            }
            raise

        self._status = {
            "state": "ready",
            "message": f"Connected to collection `{config.collection_name}`.",
            "last_error": None,
        }

    @property
    def is_ready(self) -> bool:
        return self._rag is not None

    def get_config_dict(self) -> Dict[str, Any]:
        config = self._config or MultimodalConfig()
        data = asdict(config)
        api_key = data.get("api_key") or ""
        data["api_key"] = ""
        data["api_key_configured"] = bool(api_key)
        timeout = data.get("http_timeout")
        if timeout is not None:
            data["http_timeout"] = {
                "total": getattr(timeout, "total", None),
                "connect": getattr(timeout, "connect", None),
                "sock_read": getattr(timeout, "sock_read", None),
            }
        return data

    def get_status(self) -> Dict[str, Any]:
        return dict(self._status)

    def add_document(self, **kwargs):
        self._ensure_ready()
        return self._run(self._rag.add_document(**kwargs))

    def add_documents_batch(self, documents: List[Dict[str, Any]], max_concurrent: int = 4):
        self._ensure_ready()
        return self._run(self._rag.add_documents_batch(documents, max_concurrent=max_concurrent))

    def query(self, **kwargs):
        self._ensure_ready()
        return self._run(self._rag.query(**kwargs))

    def cross_modal_search(self, **kwargs):
        self._ensure_ready()
        return self._run(self._rag.cross_modal_search(**kwargs))

    def get_collection_stats(self):
        self._ensure_ready()
        return self._run(self._rag.get_collection_stats())

    def delete(self, ids: Iterable[str]):
        self._ensure_ready()
        return self._run(self._rag.kb.delete(list(ids)))

    async def _close(self):
        if self._rag is not None:
            await self._rag.close()
            self._rag = None
            self._config = None

    def close(self):
        self._run(self._close())
        self._status = {
            "state": "disconnected",
            "message": "Knowledge base is not initialized.",
            "last_error": None,
        }

    def shutdown(self):
        try:
            self.close()
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _ensure_ready(self):
        if self._rag is None:
            raise RuntimeError("知识库尚未初始化，请先在侧边栏点击初始化。")


def save_uploaded_file(uploaded_file, subdir: str) -> str:
    upload_root = Path(gettempdir()) / "multimodal_kb_ui" / subdir
    upload_root.mkdir(parents=True, exist_ok=True)

    content = uploaded_file.getvalue()
    digest = hashlib.sha1(content).hexdigest()[:12]
    file_path = upload_root / f"{digest}_{uploaded_file.name}"
    file_path.write_bytes(content)
    return str(file_path)


def save_binary_file(filename: str, content: bytes, subdir: str) -> str:
    upload_root = Path(gettempdir()) / "multimodal_kb_ui" / subdir
    upload_root.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1(content).hexdigest()[:12]
    file_path = upload_root / f"{digest}_{Path(filename).name}"
    file_path.write_bytes(content)
    return str(file_path)
