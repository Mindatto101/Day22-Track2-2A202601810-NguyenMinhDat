"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

# Gemini free-tier returns one candidate per request; one synthetic question is
# sufficient for answer relevancy and avoids tripling quota usage.
answer_relevancy.strictness = 1

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# TODO: Copy SYSTEM_V1 và SYSTEM_V2 mà bạn đã viết ở file 02_prompt_hub_ab_routing.py
SYSTEM_V1 = (
    "You are a concise AI assistant. Use only the supplied context to answer the "
    "question in 2-4 sentences. State clearly when the context does not contain "
    "enough information.\n\nContext:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an AI subject-matter expert. Read the supplied context carefully, "
    "identify the facts relevant to the question, and give a clear, well-structured "
    "answer in 3-5 sentences. Do not introduce facts that are absent from the "
    "context; explicitly note missing information.\n\nContext:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    # TODO: Retrieve documents từ retriever
    docs = retriever.invoke(question)

    # TODO: Tạo contexts là danh sách page_content (KHÔNG ghép chuỗi ở đây)
    # Gợi ý: contexts = [doc.page_content for doc in docs]
    contexts = [doc.page_content for doc in docs]

    # TODO: Ghép contexts thành 1 string để truyền vào {context} của prompt
    ctx_str = "\n\n".join(contexts)

    # TODO: Chạy chain (prompt | llm | StrOutputParser()).invoke(...)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": ctx_str,
        "question": question,
    })

    # TODO: Trả về dict với answer và contexts (list)
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    cache_dir = Path(__file__).parent.parent / ".cache" / "ragas"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"outputs_{prompt_version}.json"
    if cache_path.exists():
        results = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"♻️ Đã nạp checkpoint {prompt_version}: {len(results)}/50 câu")
    else:
        results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS[len(results):], len(results) + 1):
        # TODO: Gọi run_rag() cho câu hỏi hiện tại
        out = run_rag(retriever, llm, prompt, qa["question"])

        # TODO: Append vào results dict với 4 keys
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        cache_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    # TODO: Tạo list các SingleTurnSample từ rag_results
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    # TODO: Wrap thành EvaluationDataset và trả về
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    cache_dir = Path(__file__).parent.parent / ".cache" / "ragas"
    cache_dir.mkdir(parents=True, exist_ok=True)
    score_path = cache_dir / f"scores_{version}.json"
    if score_path.exists():
        print(f"♻️ Đã nạp checkpoint điểm RAGAS {version}")
        return json.loads(score_path.read_text(encoding="utf-8"))

    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    # TODO: Tạo EvaluationDataset từ rag_results
    dataset = build_ragas_dataset(rag_results)

    metric_items = [
        ("faithfulness", faithfulness),
        ("answer_relevancy", answer_relevancy),
        ("context_recall", context_recall),
        ("context_precision", context_precision),
    ]
    models = config.RAGAS_EVALUATOR_MODELS
    if len(models) < len(metric_items):
        models = (models * len(metric_items))[:len(metric_items)]

    emb_eval = get_embeddings()
    scores = {}

    def evaluate_metric(key, metric, model_name):
        metric_path = cache_dir / f"metric_{version}_{key}.json"
        if metric_path.exists():
            value = float(json.loads(metric_path.read_text(encoding="utf-8"))["score"])
            print(f"♻️ {version}/{key}: {value:.4f}")
            return key, value

        print(f"▶ {version}/{key} dùng {model_name}")
        # A single RAGAS evaluation keeps its asyncio lifecycle intact. A single
        # worker and conservative RPM are needed for Gemini's 15-RPM free tier.
        safe_rpm = 6 if key == "answer_relevancy" else 10
        result = evaluate(
            dataset,
            metrics=[metric],
            llm=get_llm(
                "gemini",
                temperature=0,
                model=model_name,
                requests_per_minute=safe_rpm,
            ),
            embeddings=emb_eval,
            run_config=RunConfig(
                timeout=900,
                max_retries=8,
                max_wait=60,
                max_workers=1,
            ),
            show_progress=False,
        )
        raw = [value for value in result[key] if value is not None]
        if not raw:
            raise RuntimeError(f"RAGAS không trả điểm hợp lệ cho {version}/{key}")
        value = float(np.mean(raw))
        metric_path.write_text(
            json.dumps({"model": model_name, "score": value}, indent=2),
            encoding="utf-8",
        )
        print(f"✅ {version}/{key}: {value:.4f}")
        return key, value

    # On Windows, repeated RAGAS evaluate() calls in worker threads can close
    # the gRPC async event loop. Execute metrics in the main thread instead.
    # Per-sample checkpoints make this safe to resume after a quota retry.
    for index, (key, metric) in enumerate(metric_items):
        key, value = evaluate_metric(key, metric, models[index])
        scores[key] = value

    score_path.write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # TODO: Tạo vectorstore
    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # TODO: Lưu báo cáo vào data/ragas_report.json
    cache_dir = Path(__file__).parent.parent / ".cache" / "ragas"
    def evaluator_model(version: str):
        path = cache_dir / f"metric_{version}_faithfulness.json"
        return json.loads(path.read_text(encoding="utf-8")).get("model") if path.exists() else None

    report = {
        "sample_count_per_prompt": len(QA_PAIRS),
        "evaluation_provider": config.PROVIDER,
        "evaluation_models": {"v1": evaluator_model("v1"), "v2": evaluator_model("v2")},
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "analysis": (
            "V1 favors concise answers while V2 favors structured expert answers. "
            "Compare faithfulness and answer relevancy together: the stronger prompt "
            "is the one that improves relevance without adding unsupported claims."
        ),
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    # TODO: Ghi report vào file bằng json.dumps hoặc json.dump
    # Gợi ý: report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    report_path.write_text(report_json, encoding="utf-8")

    evidence_path = Path(__file__).parent.parent / "evidence" / "03_ragas_report.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(report_json, encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
