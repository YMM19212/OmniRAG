import ast
import asyncio
import json
import mimetypes
from io import BytesIO
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from multimodal_kb import MultimodalConfig
from ui_backend import StreamlitKBClient, save_binary_file


def ok(data: Any = None, message: str = "OK") -> Dict[str, Any]:
    return {
        "success": True,
        "message": message,
        "data": data,
        "error": None,
    }


def fail(message: str, error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "success": False,
        "message": message,
        "data": None,
        "error": error or message,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = StreamlitKBClient()
    app.state.kb_client = client
    yield
    client.shutdown()


app = FastAPI(title="OmniRAG API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_CONFIG = MultimodalConfig()
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_ROOT = Path(gettempdir()) / "multimodal_kb_ui"


def clean_optional_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def clean_or_default(value: Optional[str], fallback: str) -> str:
    if value is None:
        return fallback
    stripped = value.strip()
    return stripped or fallback


class InitializeRequest(BaseModel):
    milvus_uri: Optional[str] = Field(default_factory=lambda: DEFAULT_CONFIG.milvus_uri)
    milvus_host: Optional[str] = Field(default_factory=lambda: DEFAULT_CONFIG.milvus_host)
    milvus_port: Optional[int] = Field(default_factory=lambda: DEFAULT_CONFIG.milvus_port)
    collection_name: str = Field(default_factory=lambda: DEFAULT_CONFIG.collection_name)
    embedding_api_url: str = Field(default_factory=lambda: DEFAULT_CONFIG.embedding_api_url)
    model_name: str = Field(default_factory=lambda: DEFAULT_CONFIG.model_name)
    api_key: str = ""
    max_concurrent_embeds: int = Field(default_factory=lambda: DEFAULT_CONFIG.max_concurrent_embeds)
    enable_deduplication: bool = Field(default_factory=lambda: DEFAULT_CONFIG.enable_deduplication)
    dedup_mode: str = Field(default_factory=lambda: DEFAULT_CONFIG.dedup_mode)
    similarity_threshold: float = Field(default_factory=lambda: DEFAULT_CONFIG.similarity_threshold)


class SearchRequest(BaseModel):
    text: Optional[str] = None
    top_k: int = 5
    distance_threshold: float = 0.5
    filter_modality: Optional[str] = None


class HybridSearchRequest(BaseModel):
    text: Optional[str] = None
    target_modality: Optional[str] = None
    top_k: int = 5


class DeleteDocumentsRequest(BaseModel):
    ids: List[str] = Field(default_factory=list)


def get_client() -> StreamlitKBClient:
    return app.state.kb_client


def parse_json_field(raw_value: Optional[str], default: Any) -> Any:
    if raw_value is None or not raw_value.strip():
        return default
    return json.loads(raw_value)


def parse_structured_field(raw_value: Any, default: Any) -> Any:
    if raw_value is None:
        return default
    if isinstance(raw_value, (dict, list)):
        return raw_value
    if not isinstance(raw_value, str):
        return default

    stripped = raw_value.strip()
    if not stripped:
        return default

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except Exception:
            continue
    return default


def parse_cap_seg_field(raw_value: Any) -> Tuple[Optional[str], List[str]]:
    parsed = parse_structured_field(raw_value, {})
    if not isinstance(parsed, dict):
        return None, []

    global_caption = parsed.get("global_caption")
    local_captions = parsed.get("local_caption") or []
    if isinstance(local_captions, str):
        local_captions = [local_captions]
    if not isinstance(local_captions, list):
        local_captions = []

    text = global_caption.strip() if isinstance(global_caption, str) and global_caption.strip() else None
    cleaned_locals = [item.strip() for item in local_captions if isinstance(item, str) and item.strip()]
    return text, cleaned_locals


async def download_binary(session: aiohttp.ClientSession, url: str) -> bytes:
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.read()


def build_parquet_metadata(
    index: int,
    source_url: str,
    local_captions: List[str],
    seg_info: Any,
    source_file: str,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "source": "parquet_dataset",
        "source_row_index": index,
        "source_url": source_url,
        "source_file": source_file,
    }
    if local_captions:
        metadata["local_captions"] = local_captions
    if seg_info is not None:
        metadata["seg_info"] = seg_info
    return metadata


async def import_parquet_records(
    parquet_name: str,
    parquet_bytes: bytes,
    max_rows: Optional[int],
    max_concurrent: int,
    store_image_base64: bool,
    skip_duplicate: bool,
) -> Tuple[List[str], int, List[str]]:
    dataframe = pd.read_parquet(BytesIO(parquet_bytes))
    candidates: List[Tuple[int, str, str, List[str], Any]] = []

    for index, row in dataframe.iterrows():
        text, local_captions = parse_cap_seg_field(row.get("cap_seg"))
        url = row.get("url")
        if not text or not isinstance(url, str) or not url.strip():
            continue

        seg_info = parse_structured_field(row.get("seg_info"), None)
        candidates.append((index, url.strip(), text, local_captions, seg_info))
        if max_rows is not None and len(candidates) >= max_rows:
            break

    if not candidates:
        raise ValueError("没有从 parquet 中解析到可导入的图文对。")

    semaphore = asyncio.Semaphore(max(1, max_concurrent))
    timeout = aiohttp.ClientTimeout(total=120, connect=20)
    errors: List[str] = []
    documents: List[Dict[str, Any]] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def process_one(record: Tuple[int, str, str, List[str], Any]):
            index, url, text, local_captions, seg_info = record
            async with semaphore:
                try:
                    content = await download_binary(session, url)
                    image_path = save_binary_file(Path(url).name or f"row_{index}.bin", content, "images")
                    documents.append(
                        {
                            "text": text,
                            "image_path": image_path,
                            "metadata": build_parquet_metadata(
                                index=index,
                                source_url=url,
                                local_captions=local_captions,
                                seg_info=seg_info,
                                source_file=parquet_name,
                            ),
                            "store_image_base64": store_image_base64,
                            "extract_thumbnail": False,
                            "skip_duplicate": skip_duplicate,
                        }
                    )
                except Exception as exc:
                    errors.append(f"row {index}: {exc}")

        await asyncio.gather(*[process_one(record) for record in candidates])

    if not documents:
        raise ValueError("parquet 中的样本未能成功下载图片，请检查 URL 是否可访问。")

    return get_client().add_documents_batch(documents, max_concurrent=max(1, max_concurrent)), len(candidates), errors


def resolve_media_path(raw_path: str) -> Path:
    try:
        resolved = Path(raw_path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Media file not found.") from exc

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Media file not found.")

    allowed_roots = (
        PROJECT_ROOT.resolve(),
        UPLOAD_ROOT.resolve(),
    )
    if not any(root == resolved or root in resolved.parents for root in allowed_roots):
        raise HTTPException(status_code=403, detail="Media path is outside allowed roots.")

    return resolved


@app.get("/api/health")
def health():
    return ok({"service": "OmniRAG API"})


@app.get("/api/media")
def get_media(path: str = Query(..., min_length=1)):
    media_path = resolve_media_path(path)
    media_type, _ = mimetypes.guess_type(media_path.name)
    return FileResponse(media_path, media_type=media_type or "application/octet-stream")


@app.get("/api/kb/status")
def get_status():
    client = get_client()
    status = client.get_status()
    status["ready"] = client.is_ready
    return ok(status)


@app.get("/api/kb/config")
def get_config():
    return ok(get_client().get_config_dict())


@app.get("/api/kb/stats")
def get_stats():
    client = get_client()
    try:
        return ok(client.get_collection_stats())
    except Exception as exc:
        return fail("Unable to read collection stats.", str(exc))


@app.post("/api/kb/initialize")
def initialize(payload: InitializeRequest):
    client = get_client()
    try:
        default_config = MultimodalConfig()
        config = MultimodalConfig(
            milvus_uri=clean_optional_str(payload.milvus_uri),
            milvus_host=clean_optional_str(payload.milvus_host) or default_config.milvus_host,
            milvus_port=payload.milvus_port if payload.milvus_port is not None else default_config.milvus_port,
            collection_name=clean_or_default(payload.collection_name, default_config.collection_name),
            embedding_api_url=clean_or_default(payload.embedding_api_url, default_config.embedding_api_url),
            model_name=clean_or_default(payload.model_name, default_config.model_name),
            api_key=clean_or_default(payload.api_key, default_config.api_key),
            max_concurrent_embeds=payload.max_concurrent_embeds,
            enable_deduplication=payload.enable_deduplication,
            dedup_mode=payload.dedup_mode,
            similarity_threshold=payload.similarity_threshold,
        )
        client.initialize(config)
        return ok(
            {
                "status": client.get_status(),
                "config": client.get_config_dict(),
                "stats": client.get_collection_stats(),
            },
            "Knowledge base initialized successfully.",
        )
    except Exception as exc:
        return fail("Knowledge base initialization failed.", str(exc))


@app.post("/api/kb/documents")
async def create_document(
    text: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(default="{}"),
    store_image_base64: bool = Form(default=False),
    extract_thumbnail: bool = Form(default=True),
    skip_duplicate: bool = Form(default=True),
    image: Optional[UploadFile] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
):
    client = get_client()
    try:
        parsed_metadata = parse_json_field(metadata, {})
        image_path = None
        video_path = None

        if image is not None:
            image_path = save_binary_file(image.filename, await image.read(), "images")
        if video is not None:
            video_path = save_binary_file(video.filename, await video.read(), "videos")

        doc_id = client.add_document(
            text=text or None,
            image_path=image_path,
            video_path=video_path,
            metadata=parsed_metadata,
            store_image_base64=store_image_base64,
            extract_thumbnail=extract_thumbnail,
            skip_duplicate=skip_duplicate,
        )
        return ok({"id": doc_id}, "Document added successfully.")
    except Exception as exc:
        return fail("Failed to add document.", str(exc))


@app.post("/api/kb/documents/batch")
async def create_documents_batch(
    files: List[UploadFile] = File(default_factory=list),
    common_text: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(default="{}"),
    store_image_base64: bool = Form(default=False),
    extract_thumbnail: bool = Form(default=True),
    skip_duplicate: bool = Form(default=True),
    max_concurrent: int = Form(default=4),
):
    client = get_client()
    try:
        parsed_metadata = parse_json_field(metadata, {})
        documents = []

        for uploaded in files:
            content = await uploaded.read()
            filename = uploaded.filename or "upload.bin"
            lowered = filename.lower()
            is_video = lowered.endswith((".mp4", ".mov", ".avi", ".mkv"))
            saved_path = save_binary_file(filename, content, "videos" if is_video else "images")
            documents.append(
                {
                    "text": common_text or None,
                    "image_path": None if is_video else saved_path,
                    "video_path": saved_path if is_video else None,
                    "metadata": parsed_metadata,
                    "store_image_base64": store_image_base64,
                    "extract_thumbnail": extract_thumbnail,
                    "skip_duplicate": skip_duplicate,
                }
            )

        ids = client.add_documents_batch(documents, max_concurrent=max_concurrent)
        return ok({"ids": ids, "count": len(ids)}, "Batch import completed.")
    except Exception as exc:
        return fail("Failed to batch import documents.", str(exc))


@app.post("/api/kb/documents/parquet")
async def create_documents_from_parquet(
    parquet: UploadFile = File(...),
    max_rows: Optional[int] = Form(default=None),
    store_image_base64: bool = Form(default=False),
    skip_duplicate: bool = Form(default=True),
    max_concurrent: int = Form(default=4),
):
    client = get_client()
    try:
        if not parquet.filename or not parquet.filename.lower().endswith(".parquet"):
            raise ValueError("请上传 .parquet 文件。")

        if not client.is_ready:
            raise RuntimeError("知识库尚未初始化，请先在设置页完成初始化。")
        content = await parquet.read()
        ids, parsed_rows, errors = await import_parquet_records(
            parquet_name=parquet.filename,
            parquet_bytes=content,
            max_rows=max_rows,
            max_concurrent=max_concurrent,
            store_image_base64=store_image_base64,
            skip_duplicate=skip_duplicate,
        )
        return ok(
            {
                "ids": ids,
                "count": len(ids),
                "parsed_rows": parsed_rows,
                "failed_rows": len(errors),
                "errors": errors[:20],
            },
            "Parquet import completed.",
        )
    except Exception as exc:
        return fail("Failed to import parquet dataset.", str(exc))


@app.post("/api/kb/search")
async def search(
    payload: str = Form(...),
    image: Optional[UploadFile] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
):
    client = get_client()
    try:
        parsed_payload = SearchRequest.model_validate_json(payload)
        image_path = None
        video_path = None
        if image is not None:
            image_path = save_binary_file(image.filename, await image.read(), "queries")
        if video is not None:
            video_path = save_binary_file(video.filename, await video.read(), "queries")

        results = client.query(
            text=parsed_payload.text,
            image_path=image_path,
            video_path=video_path,
            top_k=parsed_payload.top_k,
            distance_threshold=parsed_payload.distance_threshold,
            filter_modality=parsed_payload.filter_modality,
        )
        return ok(results, "Search completed.")
    except Exception as exc:
        return fail("Search failed.", str(exc))


@app.post("/api/kb/search/hybrid")
async def hybrid_search(
    payload: str = Form(...),
    image: Optional[UploadFile] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
):
    client = get_client()
    try:
        parsed_payload = HybridSearchRequest.model_validate_json(payload)
        image_path = None
        video_path = None
        if image is not None:
            image_path = save_binary_file(image.filename, await image.read(), "queries")
        if video is not None:
            video_path = save_binary_file(video.filename, await video.read(), "queries")

        results = client.cross_modal_search(
            text=parsed_payload.text,
            image_path=image_path,
            video_path=video_path,
            target_modality=parsed_payload.target_modality,
            top_k=parsed_payload.top_k,
        )
        return ok(results, "Hybrid search completed.")
    except Exception as exc:
        return fail("Hybrid search failed.", str(exc))


@app.delete("/api/kb/documents")
def delete_documents(payload: DeleteDocumentsRequest):
    client = get_client()
    try:
        client.delete(payload.ids)
        stats = client.get_collection_stats()
        return ok({"deleted_ids": payload.ids, "stats": stats}, "Records deleted successfully.")
    except Exception as exc:
        return fail("Failed to delete records.", str(exc))
