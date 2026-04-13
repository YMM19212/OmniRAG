import argparse
import ast
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import aiohttp
import pandas as pd
from pymilvus import utility

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multimodal_kb import AsyncMultimodalRAGSystem, MultimodalConfig


DEFAULT_LLM_API_BASE = "http://172.17.80.23:18088/llm_api_yiqi3i/v1"
DEFAULT_LLM_MODEL = "Qwen3-30B-A3B-Instruct-2507"
DEFAULT_LLM_API_KEY = "sk123"


@dataclass
class SourceRecord:
    source_id: str
    url: str
    text: str
    local_captions: List[str]


@dataclass
class EvalSample:
    sample_id: str
    source_id: str
    group_id: str
    variant_type: str
    modality: str
    text: str
    image_url: Optional[str]
    image_path: Optional[str]
    expected_duplicate: bool
    line_name: str


@dataclass
class EvalResult:
    threshold: float
    line_name: str
    total: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dedup evaluation for multimodal KB.")
    parser.add_argument("--parquet-path", required=True, help="Path to parquet dataset.")
    parser.add_argument("--output-dir", default="", help="Directory to write evaluation artifacts.")
    parser.add_argument("--collection-name", default="dedup_eval", help="Base Milvus collection name.")
    parser.add_argument("--max-samples", type=int, default=20, help="Number of source rows to sample.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--download-images", dest="download_images", action="store_true", default=True)
    parser.add_argument("--skip-image-download", dest="download_images", action="store_false")
    parser.add_argument("--image-cache-dir", default="data/eval_images", help="Local image cache directory.")
    parser.add_argument("--threshold-min", type=float, default=0.70)
    parser.add_argument("--threshold-max", type=float, default=0.98)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--milvus-uri", default="./data/multimodal_kb.db")
    parser.add_argument("--embedding-api-url", default="https://api.jina.ai/v1/embeddings")
    parser.add_argument("--embedding-model", default="jina-embeddings-v4")
    parser.add_argument("--embedding-api-key", default=os.getenv("JINA_API_KEY", ""))
    parser.add_argument("--embedding-task", default="text-matching")
    parser.add_argument("--llm-api-base", default=os.getenv("DEDUP_LLM_API_BASE", DEFAULT_LLM_API_BASE))
    parser.add_argument("--llm-api-key", default=os.getenv("DEDUP_LLM_API_KEY", DEFAULT_LLM_API_KEY))
    parser.add_argument("--llm-model", default=os.getenv("DEDUP_LLM_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument("--llm-supports-json", action="store_true", default=True)
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM rewrite generation.")
    return parser.parse_args()


def build_output_dir(raw_output_dir: str) -> Path:
    if raw_output_dir:
        output_dir = Path(raw_output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data") / f"dedup_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_cap_seg(raw_value: object) -> Tuple[Optional[str], List[str]]:
    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
        return None, []

    parsed = raw_value
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return None, []
        try:
            parsed = ast.literal_eval(raw_value)
        except Exception:
            try:
                parsed = json.loads(raw_value)
            except Exception:
                return None, []

    if not isinstance(parsed, dict):
        return None, []

    global_caption = parsed.get("global_caption")
    local_captions = parsed.get("local_caption") or []
    if isinstance(local_captions, str):
        local_captions = [local_captions]

    clean_global = normalize_text(global_caption) if isinstance(global_caption, str) else None
    clean_locals = [normalize_text(item) for item in local_captions if isinstance(item, str) and normalize_text(item)]
    return clean_global, clean_locals


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    return text.strip()


def load_source_records(parquet_path: Path, max_samples: int, sample_seed: int) -> List[SourceRecord]:
    df = pd.read_parquet(parquet_path)
    records: List[SourceRecord] = []

    for index, row in df.iterrows():
        text, local_captions = parse_cap_seg(row.get("cap_seg"))
        url = row.get("url")
        if not text or not isinstance(url, str) or not url:
            continue

        records.append(
            SourceRecord(
                source_id=str(index),
                url=url,
                text=text,
                local_captions=local_captions,
            )
        )

    if max_samples and len(records) > max_samples:
        rng = random.Random(sample_seed)
        records = rng.sample(records, max_samples)

    return records


def split_sentences(text: str) -> List[str]:
    chunks = [chunk for chunk in re.split(r"[，。；！？]", text) if chunk]
    return chunks if chunks else [text]


def apply_synonym_rules(text: str) -> str:
    replacements = {
        "这是一张": "这是一个",
        "展示": "呈现",
        "照片": "图像",
        "图片": "图像",
        "背景": "画面背景",
        "可以看到": "能够看到",
        "给人一种": "呈现出",
        "非常": "相当",
        "具有": "带有",
        "特写": "近景",
    }
    updated = text
    for source, target in replacements.items():
        if source in updated:
            updated = updated.replace(source, target, 1)
    return updated


def generate_rule_rewrites(record: SourceRecord) -> List[str]:
    variants: List[str] = []
    base = record.text
    sentences = split_sentences(base)

    if len(sentences) > 1:
        variants.append("，".join(reversed(sentences)) + "。")

    variants.append(apply_synonym_rules(base))

    if record.local_captions:
        local_text = normalize_text(record.local_captions[0])
        if local_text:
            variants.append(f"{local_text}整体来看，{base}")

    deduped: List[str] = []
    for item in variants:
        normalized = normalize_text(item)
        if normalized and normalized != base and normalized not in deduped:
            deduped.append(normalized)
    return deduped[:2]


def char_bigram_set(text: str) -> set:
    normalized = normalize_text(text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index:index + 2] for index in range(len(normalized) - 1)}


def find_hard_negative_map(records: List[SourceRecord]) -> Dict[str, SourceRecord]:
    signatures = {record.source_id: char_bigram_set(record.text) for record in records}
    result: Dict[str, SourceRecord] = {}

    for record in records:
        best_match = None
        best_score = -1.0
        current = signatures[record.source_id]
        for candidate in records:
            if candidate.source_id == record.source_id:
                continue
            other = signatures[candidate.source_id]
            union = len(current | other)
            if union == 0:
                continue
            score = len(current & other) / union
            if score > best_score:
                best_score = score
                best_match = candidate
        if best_match is not None:
            result[record.source_id] = best_match

    return result


async def fetch_llm_variants(
    session: aiohttp.ClientSession,
    api_base: str,
    api_key: str,
    model: str,
    supports_json: bool,
    text: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    endpoint = api_base.rstrip("/") + "/chat/completions"
    system_prompt = (
        "你是一个中文多模态知识库评测数据生成器。"
        "请基于给定文本输出一个强同义改写版本，以及一个语义接近但不应视为同一内容的 hard negative。"
        "hard negative 需要保持题材接近，但对象、场景或核心事实必须不同。"
        "只输出 JSON。"
    )
    user_prompt = (
        f"原始文本：{text}\n"
        "返回 JSON，字段为 duplicate_rewrite 和 hard_negative。"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.6,
    }
    if supports_json:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        async with session.post(endpoint, json=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
    except Exception as exc:
        return None, None, f"llm_request_failed: {exc}"

    try:
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
    except Exception as exc:
        return None, None, f"llm_parse_failed: {exc}"

    duplicate_rewrite = normalize_text(parsed.get("duplicate_rewrite")) if isinstance(parsed, dict) else ""
    hard_negative = normalize_text(parsed.get("hard_negative")) if isinstance(parsed, dict) else ""
    return duplicate_rewrite or None, hard_negative or None, None


async def generate_llm_variants(
    records: List[SourceRecord],
    api_base: str,
    api_key: str,
    model: str,
    supports_json: bool,
    disabled: bool,
) -> Tuple[Dict[str, Dict[str, Optional[str]]], List[str]]:
    if disabled or not api_base or not model:
        return {}, ["llm_disabled"]

    errors: List[str] = []
    results: Dict[str, Dict[str, Optional[str]]] = {}
    timeout = aiohttp.ClientTimeout(total=60)
    semaphore = asyncio.Semaphore(4)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def process(record: SourceRecord):
            async with semaphore:
                duplicate_rewrite, hard_negative, error = await fetch_llm_variants(
                    session=session,
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    supports_json=supports_json,
                    text=record.text,
                )
                if error:
                    errors.append(f"{record.source_id}:{error}")
                results[record.source_id] = {
                    "duplicate_rewrite": duplicate_rewrite,
                    "hard_negative": hard_negative,
                }

        await asyncio.gather(*[process(record) for record in records])

    return results, errors


async def download_images(records: List[SourceRecord], cache_dir: Path) -> Dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=8)
    semaphore = asyncio.Semaphore(8)
    downloaded: Dict[str, str] = {}

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def fetch(record: SourceRecord):
            suffix = Path(record.url).suffix or ".jpg"
            filename = hashlib.sha1(record.url.encode("utf-8")).hexdigest() + suffix
            target = cache_dir / filename
            if target.exists():
                downloaded[record.source_id] = str(target)
                return

            async with semaphore:
                try:
                    async with session.get(record.url) as response:
                        response.raise_for_status()
                        content = await response.read()
                    target.write_bytes(content)
                    downloaded[record.source_id] = str(target)
                except Exception:
                    return

        await asyncio.gather(*[fetch(record) for record in records])

    return downloaded


def build_eval_samples(
    records: List[SourceRecord],
    hard_negative_map: Dict[str, SourceRecord],
    llm_variants: Dict[str, Dict[str, Optional[str]]],
    image_paths: Dict[str, str],
) -> List[EvalSample]:
    samples: List[EvalSample] = []

    for record in records:
        group_id = f"group_{record.source_id}"
        image_path = image_paths.get(record.source_id)

        samples.append(EvalSample(
            sample_id=f"text_original_{record.source_id}",
            source_id=record.source_id,
            group_id=group_id,
            variant_type="original",
            modality="text",
            text=record.text,
            image_url=None,
            image_path=None,
            expected_duplicate=False,
            line_name="text",
        ))

        if image_path:
            samples.append(EvalSample(
                sample_id=f"mm_original_{record.source_id}",
                source_id=record.source_id,
                group_id=group_id,
                variant_type="original",
                modality="multimodal",
                text=record.text,
                image_url=record.url,
                image_path=image_path,
                expected_duplicate=False,
                line_name="multimodal",
            ))

        for index, rewrite in enumerate(generate_rule_rewrites(record), start=1):
            samples.append(EvalSample(
                sample_id=f"text_rule_{record.source_id}_{index}",
                source_id=record.source_id,
                group_id=group_id,
                variant_type="rule_rewrite",
                modality="text",
                text=rewrite,
                image_url=None,
                image_path=None,
                expected_duplicate=True,
                line_name="text",
            ))

            if image_path:
                samples.append(EvalSample(
                    sample_id=f"mm_rule_{record.source_id}_{index}",
                    source_id=record.source_id,
                    group_id=group_id,
                    variant_type="rule_rewrite",
                    modality="multimodal",
                    text=rewrite,
                    image_url=record.url,
                    image_path=image_path,
                    expected_duplicate=True,
                    line_name="multimodal",
                ))

        llm_result = llm_variants.get(record.source_id, {})
        duplicate_rewrite = llm_result.get("duplicate_rewrite")
        hard_negative = llm_result.get("hard_negative")

        if duplicate_rewrite:
            samples.append(EvalSample(
                sample_id=f"text_llm_{record.source_id}",
                source_id=record.source_id,
                group_id=group_id,
                variant_type="llm_rewrite",
                modality="text",
                text=duplicate_rewrite,
                image_url=None,
                image_path=None,
                expected_duplicate=True,
                line_name="text",
            ))
            if image_path:
                samples.append(EvalSample(
                    sample_id=f"mm_llm_{record.source_id}",
                    source_id=record.source_id,
                    group_id=group_id,
                    variant_type="llm_rewrite",
                    modality="multimodal",
                    text=duplicate_rewrite,
                    image_url=record.url,
                    image_path=image_path,
                    expected_duplicate=True,
                    line_name="multimodal",
                ))

        hard_negative_record = hard_negative_map.get(record.source_id)
        fallback_hard_negative = hard_negative_record.text if hard_negative_record else ""
        hard_negative_text = hard_negative or fallback_hard_negative
        if hard_negative_text and hard_negative_record:
            is_fallback_neighbor = not hard_negative
            negative_variant_type = "fallback_neighbor" if is_fallback_neighbor else "hard_negative"
            expected_duplicate = is_fallback_neighbor
            samples.append(EvalSample(
                sample_id=f"text_hn_{record.source_id}",
                source_id=hard_negative_record.source_id,
                group_id=f"group_{hard_negative_record.source_id}",
                variant_type=negative_variant_type,
                modality="text",
                text=hard_negative_text,
                image_url=None,
                image_path=None,
                expected_duplicate=expected_duplicate,
                line_name="text",
            ))

            hard_negative_image = image_paths.get(hard_negative_record.source_id)
            if hard_negative_image:
                samples.append(EvalSample(
                    sample_id=f"mm_hn_{record.source_id}",
                    source_id=hard_negative_record.source_id,
                    group_id=f"group_{hard_negative_record.source_id}",
                    variant_type=negative_variant_type,
                    modality="multimodal",
                    text=hard_negative_text,
                    image_url=hard_negative_record.url,
                    image_path=hard_negative_image,
                    expected_duplicate=expected_duplicate,
                    line_name="multimodal",
                ))

    return samples


def order_samples(samples: Iterable[EvalSample]) -> List[EvalSample]:
    originals = [sample for sample in samples if sample.variant_type == "original"]
    variants = [sample for sample in samples if sample.variant_type != "original"]
    return originals + variants


def build_thresholds(min_value: float, max_value: float, step: float) -> List[float]:
    thresholds: List[float] = []
    current = min_value
    while current <= max_value + 1e-9:
        thresholds.append(round(current, 4))
        current += step
    return thresholds


async def evaluate_line(
    samples: List[EvalSample],
    threshold: float,
    line_name: str,
    args: argparse.Namespace,
) -> Tuple[EvalResult, List[Dict[str, object]], List[Dict[str, object]]]:
    collection_name = f"{args.collection_name}_{args.run_token}_{line_name}_{str(threshold).replace('.', '_')}"
    config = MultimodalConfig(
        milvus_uri=args.milvus_uri,
        collection_name=collection_name,
        embedding_api_url=args.embedding_api_url,
        model_name=args.embedding_model,
        api_key=args.embedding_api_key,
        embedding_task=args.embedding_task,
        enable_deduplication=True,
        dedup_mode="semantic",
        similarity_threshold=threshold,
    )

    rag = AsyncMultimodalRAGSystem(config)
    seen_ids: Dict[str, EvalSample] = {}
    failures: List[Dict[str, object]] = []
    decisions: List[Dict[str, object]] = []
    tp = fp = fn = tn = 0

    try:
        await rag.initialize()
        for sample in order_samples([item for item in samples if item.line_name == line_name]):
            result_id = await rag.add_document(
                text=sample.text,
                image_path=sample.image_path,
                metadata={
                    "sample_id": sample.sample_id,
                    "source_id": sample.source_id,
                    "group_id": sample.group_id,
                    "variant_type": sample.variant_type,
                    "line_name": sample.line_name,
                },
                skip_duplicate=True,
            )

            predicted_duplicate = result_id in seen_ids
            matched_sample = seen_ids.get(result_id)

            if predicted_duplicate and sample.expected_duplicate:
                tp += 1
            elif predicted_duplicate and not sample.expected_duplicate:
                fp += 1
            elif not predicted_duplicate and sample.expected_duplicate:
                fn += 1
            else:
                tn += 1

            if predicted_duplicate != sample.expected_duplicate:
                failures.append({
                    "threshold": threshold,
                    "line_name": line_name,
                    "sample_id": sample.sample_id,
                    "source_id": sample.source_id,
                    "group_id": sample.group_id,
                    "variant_type": sample.variant_type,
                    "text": sample.text,
                    "image_path": sample.image_path,
                    "expected_duplicate": sample.expected_duplicate,
                    "predicted_duplicate": predicted_duplicate,
                    "matched_id": result_id,
                    "matched_group_id": matched_sample.group_id if matched_sample else None,
                    "matched_sample_id": matched_sample.sample_id if matched_sample else None,
                })

            decisions.append({
                "threshold": threshold,
                "line_name": line_name,
                "variant_type": sample.variant_type,
                "expected_duplicate": sample.expected_duplicate,
                "predicted_duplicate": predicted_duplicate,
            })

            if not predicted_duplicate:
                seen_ids[result_id] = sample
    finally:
        try:
            await rag.close()
        finally:
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    result = EvalResult(
        threshold=threshold,
        line_name=line_name,
        total=tp + fp + fn + tn,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
    )

    variant_metrics: List[Dict[str, object]] = []
    for variant_type in sorted({item["variant_type"] for item in decisions}):
        scoped = [item for item in decisions if item["variant_type"] == variant_type]
        v_tp = sum(1 for item in scoped if item["expected_duplicate"] and item["predicted_duplicate"])
        v_fp = sum(1 for item in scoped if not item["expected_duplicate"] and item["predicted_duplicate"])
        v_fn = sum(1 for item in scoped if item["expected_duplicate"] and not item["predicted_duplicate"])
        v_tn = sum(1 for item in scoped if not item["expected_duplicate"] and not item["predicted_duplicate"])
        v_precision = v_tp / (v_tp + v_fp) if (v_tp + v_fp) else 0.0
        v_recall = v_tp / (v_tp + v_fn) if (v_tp + v_fn) else 0.0
        v_f1 = 2 * v_precision * v_recall / (v_precision + v_recall) if (v_precision + v_recall) else 0.0
        variant_metrics.append({
            "threshold": threshold,
            "line_name": line_name,
            "variant_type": variant_type,
            "total": len(scoped),
            "tp": v_tp,
            "fp": v_fp,
            "fn": v_fn,
            "tn": v_tn,
            "precision": round(v_precision, 6),
            "recall": round(v_recall, 6),
            "f1": round(v_f1, 6),
        })

    return result, failures, variant_metrics


def write_json(path: Path, payload: object):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


async def main_async(args: argparse.Namespace):
    parquet_path = Path(args.parquet_path)
    output_dir = build_output_dir(args.output_dir)
    args.run_token = output_dir.name

    source_records = load_source_records(
        parquet_path=parquet_path,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
    )
    if not source_records:
        raise ValueError("没有从 parquet 中解析到可用样本。")

    hard_negative_map = find_hard_negative_map(source_records)
    llm_variants, llm_errors = await generate_llm_variants(
        records=source_records,
        api_base=args.llm_api_base,
        api_key=args.llm_api_key,
        model=args.llm_model,
        supports_json=args.llm_supports_json,
        disabled=args.disable_llm,
    )

    image_paths: Dict[str, str] = {}
    if args.download_images:
        image_paths = await download_images(source_records, Path(args.image_cache_dir))

    samples = build_eval_samples(
        records=source_records,
        hard_negative_map=hard_negative_map,
        llm_variants=llm_variants,
        image_paths=image_paths,
    )

    thresholds = build_thresholds(args.threshold_min, args.threshold_max, args.threshold_step)
    metrics: List[Dict[str, object]] = []
    variant_metrics: List[Dict[str, object]] = []
    all_failures: Dict[str, List[Dict[str, object]]] = {"text": [], "multimodal": []}

    available_lines = ["text"]
    if any(sample.line_name == "multimodal" for sample in samples):
        available_lines.append("multimodal")

    for threshold in thresholds:
        for line_name in available_lines:
            result, failures, variant_rows = await evaluate_line(
                samples=samples,
                threshold=threshold,
                line_name=line_name,
                args=args,
            )
            metrics.append(asdict(result))
            variant_metrics.extend(variant_rows)
            all_failures[line_name].extend(failures)

    metrics_by_line: Dict[str, List[Dict[str, object]]] = {}
    best_by_line: Dict[str, Dict[str, object]] = {}
    for line_name in available_lines:
        line_metrics = [row for row in metrics if row["line_name"] == line_name]
        metrics_by_line[line_name] = line_metrics
        if line_metrics:
            best_by_line[line_name] = max(line_metrics, key=lambda row: (row["f1"], row["recall"], row["precision"]))

    best_failures: List[Dict[str, object]] = []
    for line_name, best_result in best_by_line.items():
        threshold = best_result["threshold"]
        for failure in all_failures[line_name]:
            if failure["threshold"] == threshold:
                best_failures.append(failure)

    sample_rows = [asdict(sample) for sample in samples]
    summary = {
        "parquet_path": str(parquet_path),
        "output_dir": str(output_dir),
        "sample_count": len(source_records),
        "eval_sample_count": len(samples),
        "lines": available_lines,
        "thresholds": thresholds,
        "best_by_line": best_by_line,
        "llm_errors": llm_errors,
        "downloaded_image_count": len(image_paths),
    }

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "threshold_metrics.csv", metrics)
    write_csv(output_dir / "variant_metrics.csv", variant_metrics)
    write_csv(output_dir / "generated_samples.csv", sample_rows)
    write_csv(output_dir / "failures_best.csv", best_failures)
    write_json(output_dir / "llm_errors.json", {"errors": llm_errors})

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
