import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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


class InitializeRequest(BaseModel):
    milvus_uri: Optional[str] = "./data/multimodal_kb.db"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "multimodal_kb"
    embedding_api_url: str = "https://api.jina.ai/v1/embeddings"
    model_name: str = "jina-embeddings-v4"
    api_key: str = ""
    max_concurrent_embeds: int = 8
    enable_deduplication: bool = True
    dedup_mode: str = "semantic"
    similarity_threshold: float = 0.95


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


@app.get("/api/health")
def health():
    return ok({"service": "OmniRAG API"})


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
        config = MultimodalConfig(
            milvus_uri=payload.milvus_uri.strip() if payload.milvus_uri else None,
            milvus_host=payload.milvus_host,
            milvus_port=payload.milvus_port,
            collection_name=payload.collection_name,
            embedding_api_url=payload.embedding_api_url,
            model_name=payload.model_name,
            api_key=payload.api_key.strip(),
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
