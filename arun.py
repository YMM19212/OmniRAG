import asyncio
import os

from multimodal_kb import (
    AsyncMilvusMultimodalKB,
    AsyncMultimodalRAGSystem,
    AsyncQwen3VLEmbedder,
    MultimodalConfig,
    create_kb_async,
)

__all__ = [
    "AsyncMilvusMultimodalKB",
    "AsyncMultimodalRAGSystem",
    "AsyncQwen3VLEmbedder",
    "MultimodalConfig",
    "create_kb_async",
]


async def main():
    """兼容旧入口：保留原有演示流程，但核心实现已迁移到 multimodal_kb/ 包。"""
    rag = await create_kb_async(
        collection_name="image_rag_demo",
        dedup_mode="semantic",
        enable_deduplication=True,
        max_concurrent_embeds=4,
    )

    try:
        print("【阶段1】开始构建图片知识库...")
        print("→ 插入文本...")
        doc_text_id = await rag.add_document(text="一只狗在草地上奔跑，阳光明媚")
        print(f"  ✓ 插入成功，ID: {doc_text_id}")

        print("→ 插入单张图片（带描述）...")
        doc_id1 = await rag.add_document(
            text="一只狗在草地上奔跑，阳光明媚",
            image_path="docs/gg.jpg",
            store_image_base64=False,
        )
        print(f"  ✓ 插入成功，ID: {doc_id1}")

        print("\n→ 批量插入图片库...")
        batch_docs = [
            {
                "image_path": "./docs/cat_01.jpg",
                "text": "一只橘猫在沙发上睡觉",
                "metadata": {"category": "animal", "tags": ["cat", "indoor"]},
            },
            {
                "image_path": "./docs/car_01.jpg",
                "text": "红色跑车在城市街道上行驶",
                "metadata": {"category": "vehicle", "tags": ["car", "sports_car", "red"]},
            },
            {
                "image_path": "./docs/food_01.jpg",
                "text": "精致的日式寿司摆盘",
                "metadata": {"category": "food", "tags": ["sushi", "japanese", "restaurant"]},
            },
            {
                "image_path": "./docs/beach_01.jpg",
                "text": "夕阳下的海滩，金色的沙滩和蓝色的海水",
                "metadata": {"category": "landscape", "tags": ["beach", "sunset", "ocean"]},
            },
        ]
        inserted_ids = await rag.add_documents_batch(batch_docs, max_concurrent=4)
        print(f"  ✓ 批量插入完成，共 {len(inserted_ids)} 张图片")

        print("\n→ 插入纯文本描述...")
        text_id = await rag.add_document(
            text="这不是图片，只是一段关于自然风光的描述文字",
            metadata={"type": "text_only", "category": "description"},
        )
        print(f"  ✓ 文本插入成功，ID: {text_id}")

        stats = await rag.get_collection_stats()
        print(f"\n【库状态】当前共有 {stats['entities_count']} 条记录")

        print("\n" + "=" * 50)
        print("【阶段2】开始多模态检索测试...")

        print("\n【检索1】文搜图：查询 '一只狗在草地上'...")
        text_results = await rag.query(
            text="一只猫在沙发上睡觉",
            distance_threshold=0.4,
            filter_modality="image",
            top_k=3,
        )
        print(f"找到 {len(text_results)} 个结果：")
        for index, hit in enumerate(text_results, 1):
            similarity = 1 - hit["distance"]
            print(f"  {index}. ID: {hit['id'][:8]}... | 相似度: {similarity:.3f}")
            print(f"     路径: {hit.get('image_path', 'N/A')}")
            if hit.get("text"):
                print(f"     描述: {hit['text'][:30]}...")

        print("\n【检索2】图搜图：用查询图片搜索相似图片...")
        query_img = "./docs/beach_01.jpg"
        if os.path.exists(query_img):
            image_results = await rag.query(image_path=query_img, top_k=3)
            print(f"找到 {len(image_results)} 个相似图片：")
            for hit in image_results:
                print(f"  - {hit.get('image_path')} (相似度: {1 - hit['distance']:.3f})")
        else:
            print("  （查询图片不存在，跳过）")

        print("\n【检索3】混合检索：文本描述 + 参考图片...")
        hybrid_results = await rag.cross_modal_search(
            text="精致的日式寿司摆盘",
            image_path="./docs/food_01.jpg",
            target_modality="image",
            top_k=3,
        )
        print(f"混合检索找到 {len(hybrid_results)} 个结果：")
        for hit in hybrid_results:
            print(f"  - ID: {hit['id'][:8]} | 模态: {hit['modality']} | 相似度: {1 - hit['distance']:.3f}")

        print("\n【检索4】语义搜索：查询 '某个文本'（返回所有类型）...")
        all_results = await rag.query(
            text="一只狗在草地上奔跑，阳光明媚",
            filter_modality=None,
            top_k=5,
        )
        print(f"找到 {len(all_results)} 条相关记录：")
        for hit in all_results:
            content_type = "图片" if hit["modality"] in ["image", "multimodal"] else "文本"
            print(f"  - [{content_type}] {hit.get('text', 'N/A')[:40]}")
    finally:
        await rag.close()
        print("\n【完成】资源已释放")


if __name__ == "__main__":
    asyncio.run(main())
