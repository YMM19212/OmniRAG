import atexit
import base64
import json
from typing import Any, Dict, List, Optional

import streamlit as st

from multimodal_kb import MultimodalConfig
from ui_backend import StreamlitKBClient, save_uploaded_file


st.set_page_config(page_title="多模态知识库", page_icon=":material/hub:", layout="wide")


def get_client() -> StreamlitKBClient:
    if "kb_client" not in st.session_state:
        st.session_state.kb_client = StreamlitKBClient()
        atexit.register(st.session_state.kb_client.shutdown)
    return st.session_state.kb_client


def get_status() -> Dict[str, Any]:
    if "kb_status" not in st.session_state:
        st.session_state.kb_status = {
            "state": "未连接",
            "message": "请先在侧边栏配置并初始化知识库。",
            "last_error": None,
        }
    return st.session_state.kb_status


def set_status(state: str, message: str, error: Optional[str] = None):
    st.session_state.kb_status = {
        "state": state,
        "message": message,
        "last_error": error,
    }


def parse_metadata(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text.strip():
        return None
    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("metadata 必须是 JSON 对象。")
    return parsed


def render_media_result(result: Dict[str, Any]):
    if result.get("image_path"):
        st.image(result["image_path"], use_container_width=True)
    elif result.get("image_base64"):
        st.image(base64.b64decode(result["image_base64"]), use_container_width=True)
    elif result.get("thumbnail"):
        st.image(base64.b64decode(result["thumbnail"]), use_container_width=True)


def render_result_card(result: Dict[str, Any], prefix: str):
    similarity = 1 - result["distance"] if result.get("distance") is not None else None
    container = st.container(border=True)
    with container:
        cols = st.columns([1.2, 2])
        with cols[0]:
            render_media_result(result)
            if result.get("video_path"):
                st.caption(f"视频: {result['video_path']}")
            elif result.get("image_path"):
                st.caption(f"图片: {result['image_path']}")
        with cols[1]:
            st.markdown(f"**ID**: `{result.get('id', '')}`")
            st.markdown(f"**模态**: `{result.get('modality', 'unknown')}`")
            if similarity is not None:
                st.markdown(f"**相似度**: `{similarity:.4f}`")
            if result.get("text"):
                st.markdown("**文本描述**")
                st.write(result["text"])
            if result.get("metadata") is not None:
                st.markdown("**Metadata**")
                st.json(result["metadata"])
            st.checkbox("标记删除", key=f"{prefix}_{result.get('id')}")


def collect_selected_ids(results: List[Dict[str, Any]], prefix: str) -> List[str]:
    return [
        result["id"]
        for result in results
        if st.session_state.get(f"{prefix}_{result.get('id')}")
    ]


def sidebar_config():
    st.sidebar.title("连接配置")
    default_config = MultimodalConfig()
    milvus_uri = st.sidebar.text_input("Milvus URI / DB Path", value=default_config.milvus_uri)
    host = st.sidebar.text_input("Milvus Host", value=default_config.milvus_host)
    port = st.sidebar.number_input("Milvus Port", min_value=1, max_value=65535, value=default_config.milvus_port)
    collection = st.sidebar.text_input("Collection Name", value=default_config.collection_name)
    embedding_url = st.sidebar.text_input("Embedding URL", value=default_config.embedding_api_url)
    model_name = st.sidebar.text_input("Embedding Model", value=default_config.model_name)
    api_key = st.sidebar.text_input("Embedding API Key", value=default_config.api_key, type="password")
    max_concurrent_embeds = st.sidebar.slider("最大并发嵌入数", min_value=1, max_value=16, value=default_config.max_concurrent_embeds)
    enable_dedup = st.sidebar.checkbox("启用去重", value=default_config.enable_deduplication)
    dedup_mode = st.sidebar.selectbox("去重模式", options=["semantic", "strict"], index=0)

    if st.sidebar.button("初始化 / 重连", use_container_width=True):
        client = get_client()
        set_status("连接中", "正在初始化知识库连接...")
        try:
            config = MultimodalConfig(
                milvus_uri=milvus_uri.strip() or None,
                milvus_host=host,
                milvus_port=int(port),
                collection_name=collection,
                embedding_api_url=embedding_url,
                model_name=model_name.strip() or default_config.model_name,
                api_key=api_key.strip(),
                max_concurrent_embeds=int(max_concurrent_embeds),
                enable_deduplication=enable_dedup,
                dedup_mode=dedup_mode,
                similarity_threshold=default_config.similarity_threshold if dedup_mode == "semantic" else 1.0,
            )
            client.initialize(config)
            set_status("已连接", f"已连接到集合 `{collection}`")
        except Exception as exc:
            set_status("初始化失败", "知识库初始化失败，请检查配置。", str(exc))


def tab_status():
    client = get_client()
    status = get_status()
    st.subheader("运行状态")
    st.write(status["message"])
    if status["last_error"]:
        st.error(status["last_error"])

    if client.is_ready:
        try:
            st.subheader("集合统计")
            st.json(client.get_collection_stats())
            st.subheader("当前配置")
            st.json(client.get_config_dict())
        except Exception as exc:
            st.error(f"读取统计失败: {exc}")
    else:
        st.info("初始化完成后，这里会显示集合统计和当前配置。")


def tab_add_single():
    st.subheader("单条入库")
    client = get_client()
    if not client.is_ready:
        st.warning("请先初始化知识库。")
        return

    text = st.text_area("文本描述", height=120, placeholder="可选：为图片或视频添加文本描述")
    metadata_text = st.text_area("Metadata JSON", value="{}", height=120)
    image_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "webp"], key="single_image")
    video_file = st.file_uploader("上传视频", type=["mp4", "mov", "avi", "mkv"], key="single_video")
    store_image_base64 = st.checkbox("存储图片 base64", value=False)
    extract_thumbnail = st.checkbox("提取视频缩略图", value=True)
    skip_duplicate = st.checkbox("跳过重复内容", value=True)

    if st.button("添加文档", type="primary", use_container_width=True):
        try:
            metadata = parse_metadata(metadata_text)
            image_path = save_uploaded_file(image_file, "images") if image_file else None
            video_path = save_uploaded_file(video_file, "videos") if video_file else None
            doc_id = client.add_document(
                text=text or None,
                image_path=image_path,
                video_path=video_path,
                metadata=metadata,
                store_image_base64=store_image_base64,
                extract_thumbnail=extract_thumbnail,
                skip_duplicate=skip_duplicate,
            )
            st.success(f"入库成功，ID: {doc_id}")
        except Exception as exc:
            st.error(f"入库失败: {exc}")


def tab_add_batch():
    st.subheader("批量入库")
    client = get_client()
    if not client.is_ready:
        st.warning("请先初始化知识库。")
        return

    files = st.file_uploader(
        "上传多个图片或视频",
        type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "avi", "mkv"],
        accept_multiple_files=True,
        key="batch_files",
    )
    common_text = st.text_area("统一文本描述", placeholder="可选：为空时只根据文件入库")
    metadata_text = st.text_area("统一 Metadata JSON", value="{}", height=120, key="batch_metadata")
    store_image_base64 = st.checkbox("批量保存图片 base64", value=False, key="batch_store_image")
    extract_thumbnail = st.checkbox("批量提取视频缩略图", value=True, key="batch_extract_thumb")
    max_concurrent = st.slider("批量并发", min_value=1, max_value=8, value=4)

    if st.button("开始批量入库", type="primary", use_container_width=True):
        if not files:
            st.warning("请先上传文件。")
            return
        try:
            metadata = parse_metadata(metadata_text)
            documents = []
            for uploaded in files:
                suffix = uploaded.name.lower()
                is_video = suffix.endswith((".mp4", ".mov", ".avi", ".mkv"))
                saved_path = save_uploaded_file(uploaded, "videos" if is_video else "images")
                documents.append({
                    "text": common_text or None,
                    "image_path": None if is_video else saved_path,
                    "video_path": saved_path if is_video else None,
                    "metadata": metadata,
                    "store_image_base64": store_image_base64,
                    "extract_thumbnail": extract_thumbnail,
                })

            ids = client.add_documents_batch(documents, max_concurrent=max_concurrent)
            st.success(f"批量入库完成，成功 {len(ids)} 条。")
            if ids:
                st.code("\n".join(ids))
        except Exception as exc:
            st.error(f"批量入库失败: {exc}")


def tab_search():
    st.subheader("基础检索")
    client = get_client()
    if not client.is_ready:
        st.warning("请先初始化知识库。")
        return

    query_text = st.text_area("查询文本", height=100)
    image_file = st.file_uploader("查询图片", type=["png", "jpg", "jpeg", "webp"], key="search_image")
    video_file = st.file_uploader("查询视频", type=["mp4", "mov", "avi", "mkv"], key="search_video")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
    distance_threshold = st.slider("距离阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    modality = st.selectbox("模态过滤", options=[None, "image", "text", "video_all", "visual"], format_func=lambda x: x or "全部")

    if st.button("执行基础检索", type="primary", use_container_width=True):
        try:
            image_path = save_uploaded_file(image_file, "queries") if image_file else None
            video_path = save_uploaded_file(video_file, "queries") if video_file else None
            results = client.query(
                text=query_text or None,
                image_path=image_path,
                video_path=video_path,
                top_k=top_k,
                distance_threshold=distance_threshold,
                filter_modality=modality,
            )
            st.session_state.last_results = results
        except Exception as exc:
            st.error(f"检索失败: {exc}")

    results = st.session_state.get("last_results", [])
    if results:
        st.caption(f"最近一次检索返回 {len(results)} 条结果")
        for result in results:
            render_result_card(result, "basic_result")


def tab_hybrid_search():
    st.subheader("混合检索")
    client = get_client()
    if not client.is_ready:
        st.warning("请先初始化知识库。")
        return

    text = st.text_area("混合查询文本", height=100, key="hybrid_text")
    image_file = st.file_uploader("参考图片", type=["png", "jpg", "jpeg", "webp"], key="hybrid_image")
    video_file = st.file_uploader("参考视频", type=["mp4", "mov", "avi", "mkv"], key="hybrid_video")
    top_k = st.slider("混合检索 Top K", min_value=1, max_value=20, value=5, key="hybrid_topk")
    target_modality = st.selectbox("目标模态", options=[None, "image", "video", "text", "visual"], format_func=lambda x: x or "全部")

    if st.button("执行混合检索", type="primary", use_container_width=True):
        try:
            image_path = save_uploaded_file(image_file, "queries") if image_file else None
            video_path = save_uploaded_file(video_file, "queries") if video_file else None
            results = client.cross_modal_search(
                text=text or None,
                image_path=image_path,
                video_path=video_path,
                target_modality=target_modality,
                top_k=top_k,
            )
            st.session_state.hybrid_results = results
        except Exception as exc:
            st.error(f"混合检索失败: {exc}")

    results = st.session_state.get("hybrid_results", [])
    if results:
        st.caption(f"最近一次混合检索返回 {len(results)} 条结果")
        for result in results:
            render_result_card(result, "hybrid_result")


def tab_manage():
    st.subheader("集合管理")
    client = get_client()
    if not client.is_ready:
        st.warning("请先初始化知识库。")
        return

    col1, col2 = st.columns(2)
    with col1:
        if st.button("刷新统计", use_container_width=True):
            try:
                st.session_state.latest_stats = client.get_collection_stats()
            except Exception as exc:
                st.error(f"刷新失败: {exc}")
    with col2:
        if st.button("清除最近结果选择", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("basic_result_") or key.startswith("hybrid_result_"):
                    st.session_state.pop(key)

    if "latest_stats" in st.session_state:
        st.json(st.session_state.latest_stats)

    delete_candidates = []
    delete_candidates.extend(collect_selected_ids(st.session_state.get("last_results", []), "basic_result"))
    delete_candidates.extend(collect_selected_ids(st.session_state.get("hybrid_results", []), "hybrid_result"))
    delete_candidates = sorted(set(delete_candidates))

    st.markdown("**待删除 ID**")
    if delete_candidates:
        st.code("\n".join(delete_candidates))
    else:
        st.caption("先在检索结果中勾选“标记删除”。")

    if st.button("删除选中结果", type="primary", use_container_width=True):
        if not delete_candidates:
            st.warning("当前没有选中的结果。")
            return
        try:
            client.delete(delete_candidates)
            st.success(f"已删除 {len(delete_candidates)} 条记录。")
            st.session_state.last_results = [r for r in st.session_state.get("last_results", []) if r["id"] not in delete_candidates]
            st.session_state.hybrid_results = [r for r in st.session_state.get("hybrid_results", []) if r["id"] not in delete_candidates]
            st.session_state.latest_stats = client.get_collection_stats()
        except Exception as exc:
            st.error(f"删除失败: {exc}")


def main():
    st.title("多模态知识库 UI")
    st.caption("面向本地测试环境的 Streamlit 前端，覆盖初始化、入库、检索和管理能力。")
    sidebar_config()

    tabs = st.tabs(["配置/状态", "单条入库", "批量入库", "基础检索", "混合检索", "管理"])
    with tabs[0]:
        tab_status()
    with tabs[1]:
        tab_add_single()
    with tabs[2]:
        tab_add_batch()
    with tabs[3]:
        tab_search()
    with tabs[4]:
        tab_hybrid_search()
    with tabs[5]:
        tab_manage()


if __name__ == "__main__":
    main()
