"""
Retrieval eval and latency benchmark for Forsa.

Runs everything that happens before the LLM call (scope detection, embedding,
FAISS search, lexical search, merge, rerank, scope filter, prompt assembly)
through api.chat.build_prompt, the same function the endpoint uses. It makes
no LLM or other network calls, so it needs no API key.

Usage:
    python eval/run_eval.py --label baseline
    python eval/run_eval.py --label tuned --batch
    python eval/run_eval.py --label noRerank --set USE_RERANKER=False

Writes eval/results/<label>.json and prints a summary.
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
# The base conda env ships TensorFlow; keep transformers from importing it.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

HIT_KS = (1, 3, 5, 10)
WARMUP_QUESTIONS = 3


def doc_slug(metadata: dict) -> str:
    """Document id of a chunk: its chunk file name without the __ChunkNNN suffix."""
    name = metadata.get("doc") or metadata.get("chunk_id") or metadata.get("file", "")
    name = re.sub(r"__Chunk\d+(\.md)?$", "", os.path.basename(name))
    return name.rstrip("_")


def unique_in_order(items):
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def percentile(values, pct):
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[idx]


def summarize_ms(values):
    return {
        "mean": round(statistics.mean(values) * 1000, 2),
        "p50": round(percentile(values, 50) * 1000, 2),
        "p95": round(percentile(values, 95) * 1000, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="name of this run, used for the results file")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--batch", action="store_true", help="also time the whole set through the batched retrieval path")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config.py setting for this run")
    args = parser.parse_args()

    import torch
    from config import config

    overrides = {}
    for item in args.set:
        key, raw = item.split("=", 1)
        if not hasattr(config, key):
            parser.error(f"unknown config key: {key}")
        overrides[key] = json.loads(raw.lower()) if raw.lower() in ("true", "false") else json.loads(raw)
        setattr(config, key, overrides[key])

    load_start = time.perf_counter()
    from api.chat import build_prompt
    from services.rag_service import rag_service
    model_load_s = time.perf_counter() - load_start

    tokenizer = rag_service.embedding_service.model.tokenizer
    questions = json.load(open(args.questions, encoding="utf-8"))["questions"]

    for q in questions[:WARMUP_QUESTIONS]:
        build_prompt(q["question"])

    rows = []
    for q in questions:
        timings = {}
        t0 = time.perf_counter()
        chunks, messages = build_prompt(q["question"], timings=timings)
        timings["total"] = time.perf_counter() - t0

        context_docs = unique_in_order(doc_slug(c.chunk.metadata or {}) for c in chunks)
        context_text = "\n".join(c.chunk.content or "" for c in chunks).lower()
        expected = {d.rstrip("_") for d in q["expectedDocs"]}
        rank = next((i + 1 for i, d in enumerate(context_docs) if d in expected), None)
        keywords = q["answerKeywords"]
        user_prompt = messages[1]["content"]

        rows.append({
            "id": q["id"],
            "lang": q["lang"],
            "inScope": bool(expected),
            "rank": rank,
            "contextChunkCount": len(chunks),
            "contextDocs": context_docs[:10],
            "contextChunks": [[doc_slug(c.chunk.metadata or {}), round(c.score, 4)] for c in chunks],
            "keywordRecall": (sum(k.lower() in context_text for k in keywords) / len(keywords)) if keywords else None,
            "promptChars": len(messages[0]["content"]) + len(user_prompt),
            "promptTokens": len(tokenizer(messages[0]["content"] + user_prompt)["input_ids"]),
            "timingsMs": {k: round(v * 1000, 2) for k, v in timings.items()},
        })

    positives = [r for r in rows if r["inScope"]]
    negatives = [r for r in rows if not r["inScope"]]

    def rate(subset, pred):
        return round(sum(1 for r in subset if pred(r)) / len(subset), 3) if subset else None

    retrieval = {}
    for name, subset in (("all", positives), ("fr", [r for r in positives if r["lang"] == "fr"]), ("ar", [r for r in positives if r["lang"] == "ar"])):
        retrieval[name] = {f"hit@{k}": rate(subset, lambda r, k=k: r["rank"] is not None and r["rank"] <= k) for k in HIT_KS}
        retrieval[name]["hitInContext"] = rate(subset, lambda r: r["rank"] is not None)
        retrieval[name]["mrr"] = round(sum(1 / r["rank"] for r in subset if r["rank"]) / len(subset), 3)
        retrieval[name]["keywordRecall"] = round(statistics.mean(r["keywordRecall"] for r in subset), 3)
        retrieval[name]["n"] = len(subset)

    stage_names = unique_in_order(k for r in rows for k in r["timingsMs"])
    latency = {
        stage: summarize_ms([r["timingsMs"][stage] / 1000 for r in rows if stage in r["timingsMs"]])
        for stage in stage_names
    }

    summary = {
        "label": args.label,
        "overrides": overrides,
        "config": {
            "useReranker": config.USE_RERANKER,
            "similarityThreshold": config.SIMILARITY_THRESHOLD,
            "topK": config.TOP_K_SEMANTIC,
            "indexVectors": rag_service.vector_store.index.ntotal,
            "device": str(rag_service.embedding_service.model.device),
            "embeddingDtype": str(next(rag_service.embedding_service.model.parameters()).dtype),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "modelLoadSeconds": round(model_load_s, 2),
        "gpuPeakMemoryMb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
        "retrieval": retrieval,
        "refusedOutOfScope": rate(negatives, lambda r: r["contextChunkCount"] == 0),
        "refusedInScope": rate(positives, lambda r: r["contextChunkCount"] == 0),
        "contextChunksMean": round(statistics.mean(r["contextChunkCount"] for r in positives), 1),
        "promptTokensMean": round(statistics.mean(r["promptTokens"] for r in rows)),
        "latencyMs": latency,
    }

    if args.batch:
        summary["batch"] = time_batch(questions, build_prompt)

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "questions": rows}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path.relative_to(ROOT)}")


def time_batch(questions, build_prompt):
    """
    Wall time for the whole question set, the way one /api/chat request with
    every question in it runs: concurrent per-question threads, and the batched
    path when the service provides one.
    """
    texts = [q["question"] for q in questions]

    async def concurrent():
        await asyncio.gather(*(asyncio.to_thread(build_prompt, t) for t in texts))

    result = {"questions": len(texts)}
    t0 = time.perf_counter()
    asyncio.run(concurrent())
    result["concurrentThreadsSeconds"] = round(time.perf_counter() - t0, 3)

    import api.chat as chat
    if hasattr(chat, "build_prompts"):
        t0 = time.perf_counter()
        chat.build_prompts(texts)
        result["batchedSeconds"] = round(time.perf_counter() - t0, 3)
    return result


if __name__ == "__main__":
    main()
