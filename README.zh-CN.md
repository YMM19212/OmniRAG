# OmniRAG

[English](./README.md) | [简体中文](./README.zh-CN.md)

OmniRAG 是一个面向文本、图像与视频的专业级多模态检索工作台，用于构建、运行和评估多模态知识库。

它将现代化 Web 控制台、Python API 服务层和基于 Milvus 的检索引擎整合在一起，形成一套本地优先的多模态知识库系统，支持数据导入、语义检索、混合检索与记录管理。

## Quick Start

### 1. 克隆仓库

```bash
git clone https://github.com/YMM19212/OmniRAG.git
cd OmniRAG
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 安装前端依赖

```bash
cd dashboard
npm install
cd ..
```

### 4. 创建 `.env` 配置文件

```bash
cp .env.example .env
```

常见配置方式：

- Docker 版 Milvus：保持 `OMNIRAG_MILVUS_URI=` 为空，并设置 `OMNIRAG_MILVUS_HOST` / `OMNIRAG_MILVUS_PORT`
- Milvus Lite：设置 `OMNIRAG_MILVUS_URI=./data/multimodal_kb.db`
- Jina 或兼容 embedding 服务：设置 `OMNIRAG_EMBEDDING_API_URL`、`OMNIRAG_EMBEDDING_MODEL_NAME`、`OMNIRAG_EMBEDDING_API_KEY`

同时保留了对 `JINA_API_KEY` 的兼容回退支持。

### 5. 如果你使用 Docker 模式，再启动 Milvus

推荐用于远程 / standalone Milvus：

```bash
cd infra/milvus
docker compose up -d
cd ../..
```

启动后端口如下：

- gRPC: `127.0.0.1:19530`
- 健康检查 / 管理端口: `127.0.0.1:9091`

如果你通过 `OMNIRAG_MILVUS_URI` 使用 Milvus Lite，可以跳过这一步。

### 6. 启动 API 服务

```bash
uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```

### 7. 启动前端

```bash
cd dashboard
npm run dev
```

访问地址：

- 前端：`http://127.0.0.1:5173`
- API：`http://127.0.0.1:8000`
- 健康检查：`http://127.0.0.1:8000/api/health`

## 为什么是 OmniRAG

很多多模态 RAG 项目只停留在 notebook 或者简单聊天界面。OmniRAG 更强调真正可操作的工程化工作台：

- 面向配置、导入、检索与记录治理的专业控制台
- 独立的 FastAPI 服务层，而不是把运行逻辑绑死在 UI 中
- 支持纯文本、图像加文本、视频加文本等索引流程
- 同时支持语义检索与跨模态混合检索
- 基于 Milvus 的向量存储与去重能力

## 功能特性

- 专业化 Web 控制台
- 中英文界面切换
- 单条导入与批量导入
- 文本、图片、视频检索
- 跨模态混合检索
- 运行状态与配置查看
- 集合统计与结果管理
- 勾选式记录删除
- 保留 Streamlit 兼容入口

## 架构

```text
OmniRAG
|- dashboard/          React + Vite + Tailwind 前端
|- api_server.py       FastAPI 服务层
|- multimodal_kb/      多模态 RAG 核心逻辑
|- ui_backend.py       UI/API 共享的同步运行桥接层
|- app.py              旧版 Streamlit 界面
`- infra/milvus/       Milvus standalone 的 Docker Compose 配置
```

### 核心技术栈

- 前端：React 19、TypeScript、Tailwind CSS、Vite
- 后端：FastAPI
- 向量数据库：Milvus
- Embedding：Jina `jina-embeddings-v4`
- 多媒体处理：OpenCV、Pillow、NumPy

## 项目结构

```text
dashboard/
multimodal_kb/
infra/milvus/
api_server.py
ui_backend.py
app.py
requirements.txt
```

## API 概览

### 知识库生命周期

- `POST /api/kb/initialize`
- `GET /api/kb/status`
- `GET /api/kb/config`
- `GET /api/kb/stats`

### 文档操作

- `POST /api/kb/documents`
- `POST /api/kb/documents/batch`
- `DELETE /api/kb/documents`

### 检索接口

- `POST /api/kb/search`
- `POST /api/kb/search/hybrid`

### 健康检查

- `GET /api/health`

## 运行模式

### 推荐模式

当前最稳定的运行组合是：

- React 控制台
- FastAPI 后端
- 通过 `.env` 配置的 Docker 版 Milvus standalone

### Streamlit 兼容模式

仓库中仍保留原始 Streamlit 界面，便于兼容和调试：

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 10188
```

## 配置说明

### 基于环境变量的配置

OmniRAG 现在会从项目根目录的 `.env` 读取运行时默认配置。后端、Streamlit 界面和 dashboard 初始化表单都会使用同一套配置源。

创建方式：

```bash
cp .env.example .env
```

关键变量包括：

- `OMNIRAG_EMBEDDING_API_URL`
- `OMNIRAG_EMBEDDING_MODEL_NAME`
- `OMNIRAG_EMBEDDING_API_KEY` 或 `JINA_API_KEY`
- `OMNIRAG_MILVUS_URI`
- `OMNIRAG_MILVUS_HOST`
- `OMNIRAG_MILVUS_PORT`
- `OMNIRAG_COLLECTION_NAME`
- `OMNIRAG_VIDEO_SAMPLE_FRAMES`
- `OMNIRAG_MAX_CONCURRENT_EMBEDS`
- `OMNIRAG_ENABLE_DEDUPLICATION`
- `OMNIRAG_DEDUP_MODE`

### Milvus Lite 与 Docker Milvus

OmniRAG 同时支持 Milvus Lite 和远程 Milvus。

使用 Milvus Lite：

- `OMNIRAG_MILVUS_URI=./data/multimodal_kb.db`

使用 Docker / 远程 Milvus：

- `OMNIRAG_MILVUS_URI=`
- `OMNIRAG_MILVUS_HOST=127.0.0.1`
- `OMNIRAG_MILVUS_PORT=19530`

### 默认集合名

```text
multimodal_kb
```

### 默认 Embedding 模型

```text
jina-embeddings-v4
```

## 开发

### 前端开发

```bash
cd dashboard
npm run dev
```

Vite 开发服务器会将 `/api` 自动代理到 `http://127.0.0.1:8000`。

### 前端构建

```bash
cd dashboard
npm run build
```

## 当前限制

- Embedding 请求依赖有效的外部 Jina API Key
- Milvus Lite 在不同操作系统和 Python 版本上的可用性并不一致
- 当前部署方式主要针对本地开发验证，尚未封装为完整生产发布形态

## 路线图

- 更完善的密钥保护与配置安全
- 更好的上传进度与长任务反馈
- 更丰富的集合管理能力
- 检索质量评估工具
- 更完善的多模态结果预览
- 可选的一体化单服务部署模式

## License

本项目基于 [MIT License](./LICENSE) 开源。
