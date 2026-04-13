# OmniRAG

**OmniRAG** is a multimodal retrieval workspace for building, operating, and evaluating knowledge bases over **text, images, and video**.

It combines a modern web console, a Python API layer, and a Milvus-backed storage engine into a practical local-first system for multimodal ingestion, semantic retrieval, hybrid search, and record management.

## Why OmniRAG?

Most RAG demos stop at a single notebook or a basic chat UI. OmniRAG is designed as an **operational surface**:

- A professional dashboard for configuration, ingestion, retrieval, and record governance
- A Python service layer that exposes reusable APIs instead of embedding logic inside the UI
- Support for **text-only**, **image + text**, and **video + text** indexing workflows
- Both **semantic search** and **hybrid cross-modal search**
- Milvus-backed storage with deduplication support

## Features

- **Professional web console**
  - Overview dashboard
  - Connection and runtime configuration
  - Single-item ingestion
  - Batch ingestion
  - Semantic search
  - Hybrid search
  - Record review and deletion

- **Multimodal knowledge ingestion**
  - Text
  - Images
  - Videos
  - Metadata payloads

- **Retrieval workflows**
  - Text-to-text
  - Text-to-image
  - Image-based retrieval
  - Video-based retrieval
  - Hybrid search across modalities

- **Operational capabilities**
  - Collection stats
  - Runtime config inspection
  - Selection-based deletion
  - Duplicate detection
  - Shared result cache across retrieval pages

- **Developer-friendly architecture**
  - React + Vite frontend
  - FastAPI backend
  - `pymilvus` integration
  - Streamlit fallback UI retained for compatibility

## Architecture

```text
OmniRAG
├─ dashboard/          # React + Vite + Tailwind frontend
├─ api_server.py       # FastAPI service layer
├─ multimodal_kb/      # Core multimodal RAG logic
├─ ui_backend.py       # Synchronous wrapper / runtime bridge
├─ app.py              # Legacy Streamlit UI
└─ infra/milvus/       # Docker Compose for Milvus standalone
```

### Core stack

- **Frontend:** React 19, TypeScript, Tailwind CSS, Vite
- **Backend:** FastAPI
- **Vector store:** Milvus
- **Embeddings:** Jina `jina-embeddings-v4`
- **Media processing:** OpenCV, Pillow, NumPy

## Project Status

OmniRAG is currently an **active local development project** focused on a production-style operator experience for multimodal RAG.

The current implementation already supports:

- API-driven initialization
- Real Milvus-backed collections
- Real document ingestion
- Real semantic and hybrid retrieval
- Dashboard-based operations

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd OmniRAG
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd dashboard
npm install
cd ..
```

### 4. Start Milvus

#### Recommended: Docker-based Milvus standalone

```bash
cd infra/milvus
docker compose up -d
cd ../..
```

Milvus will be exposed on:

- gRPC: `127.0.0.1:19530`
- health / admin: `127.0.0.1:9091`

### 5. Start the API

```bash
uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```

### 6. Start the frontend

```bash
cd dashboard
npm run dev
```

Open:

- Frontend: `http://127.0.0.1:5173`
- API: `http://127.0.0.1:8000`

## Authentication / API Keys

OmniRAG uses Jina embeddings by default.

You can provide the key in either of the following ways:

- set `JINA_API_KEY` in your environment
- enter the key in the **Connection & Settings** page in the dashboard

Example:

```bash
set JINA_API_KEY=your_jina_api_key_here
```

Without a valid embedding key, initialization or retrieval requests that depend on embeddings will fail.

## Running Modes

### Recommended mode

Use:

- **React dashboard** as the main UI
- **FastAPI** as the runtime service
- **Docker Milvus standalone** as the vector backend

This is the most stable setup across platforms.

### Streamlit fallback

The repository still includes the original Streamlit UI:

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 10188
```

This is kept for compatibility and debugging, but it is **not the primary interface** anymore.

## API Overview

The FastAPI server exposes the following endpoints:

### Knowledge base lifecycle

- `POST /api/kb/initialize`
- `GET /api/kb/status`
- `GET /api/kb/config`
- `GET /api/kb/stats`

### Document operations

- `POST /api/kb/documents`
- `POST /api/kb/documents/batch`
- `DELETE /api/kb/documents`

### Retrieval

- `POST /api/kb/search`
- `POST /api/kb/search/hybrid`

### Health

- `GET /api/health`

## Frontend Development

The frontend lives in [`dashboard/`](./dashboard).

### Dev server

```bash
cd dashboard
npm run dev
```

By default, the Vite dev server proxies `/api` to:

```text
http://127.0.0.1:8000
```

### Production build

```bash
cd dashboard
npm run build
```

## Configuration Notes

### Milvus Lite vs remote Milvus

OmniRAG supports a local database path through `pymilvus[milvus_lite]`, but in practice **platform compatibility varies**.

For example, on some Windows + Python 3.13 environments, `milvus-lite` may not be available as an installable package. In those cases, use **Docker Milvus standalone** and configure:

- `milvus_uri = null`
- `milvus_host = 127.0.0.1`
- `milvus_port = 19530`

### Default collection

The default collection name is:

```text
multimodal_kb
```

### Default embedding model

```text
jina-embeddings-v4
```

## Current UX Highlights

- Professional dashboard UI replacing the original Streamlit-only surface
- English / Chinese language toggle in the console
- Shared result review flow across retrieval pages
- Runtime configuration panel with real backend integration
- Removal of template watermarks / promo leftovers from the original dashboard template

## Known Limitations

- The current embedding integration depends on an external Jina API key
- Milvus Lite is not equally available on all operating systems / Python versions
- The legacy Streamlit app remains in the repository, but the main product surface is now the React dashboard
- API key masking in config responses should be hardened further before wider distribution

## Roadmap

- Better API key masking and secret handling
- Improved upload progress and background task UX
- Richer collection management
- Evaluation tooling for retrieval quality
- Better multimodal result previews
- Optional single-binary or single-service deployment mode

## Development Notes

If you are working locally on Windows, the most reliable stack today is:

1. Docker Milvus standalone
2. FastAPI backend
3. React dashboard

That setup has already been validated in this repository.

## License

This project currently does not define a final public license in the repository root.

If you plan to publish OmniRAG openly on GitHub, add an explicit license file before release.

## Acknowledgements

OmniRAG uses a customized dashboard foundation and replaces its original branding and demo content with a retrieval-focused operator workflow.

The current project direction is focused on building a **serious multimodal RAG workspace**, not a generic admin template.
