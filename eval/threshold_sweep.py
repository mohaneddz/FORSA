"""
Sweep RERANK_MIN_SCORE offline from one eval run that kept every reranked
candidate (RERANK_MIN_SCORE=0, MAX_SOURCES=20).

Usage:
    python eval/run_eval.py --label rerankSweep --set RERANK_MIN_SCORE=0 --set MAX_SOURCES=20
    python eval/threshold_sweep.py eval/results/rerankSweep.json
"""
import json
import sys
from pathlib import Path

THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
MAX_SOURCES = 6


def sweep(rows, questions):
    in_scope = [r for r in rows if r["inScope"]]
    out_of_scope = [r for r in rows if not r["inScope"]]
    table = []
    for threshold in THRESHOLDS:
        hits_at_1 = hits_in_context = refused_in_scope = context_sizes = 0
        for row in in_scope:
            expected = set(questions[row["id"]]["expectedDocs"])
            kept = [doc for doc, score in row["contextChunks"] if score >= threshold][:MAX_SOURCES]
            context_sizes += len(kept)
            refused_in_scope += not kept
            hits_at_1 += bool(kept) and kept[0] in expected
            hits_in_context += any(doc in expected for doc in kept)
        refused_out_of_scope = sum(
            1 for row in out_of_scope if not any(score >= threshold for _, score in row["contextChunks"])
        )
        n = len(in_scope)
        table.append({
            "threshold": threshold,
            "hit@1": round(hits_at_1 / n, 3),
            "hitInContext": round(hits_in_context / n, 3),
            "inScopeRefused": refused_in_scope,
            "outOfScopeRefused": f"{refused_out_of_scope}/{len(out_of_scope)}",
            "contextChunksMean": round(context_sizes / n, 1),
        })
    return table


def main():
    results_path = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/results/rerankSweep.json")
    rows = json.load(open(results_path, encoding="utf-8"))["questions"]
    questions = {q["id"]: q for q in json.load(open("eval/questions.json", encoding="utf-8"))["questions"]}
    top_out_of_scope = max((r["contextChunks"][0][1] for r in rows if not r["inScope"] and r["contextChunks"]), default=None)
    print(f"highest reranker score on an out-of-scope question: {top_out_of_scope}")
    print("| threshold | hit@1 | source in context | in-scope refused | out-of-scope refused | chunks sent |")
    print("|---|---|---|---|---|---|")
    for line in sweep(rows, questions):
        print(f"| {line['threshold']} | {line['hit@1']} | {line['hitInContext']} | {line['inScopeRefused']} | "
              f"{line['outOfScopeRefused']} | {line['contextChunksMean']} |")


if __name__ == "__main__":
    main()
