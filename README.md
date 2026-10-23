# Forsa

Bilingual (French / Arabic) question answering over Algérie Télécom's internal sales documentation: offer sheets ("argumentaires"), NGBSS operator guides and partner conventions. A FastAPI service retrieves passages with hybrid semantic + lexical search, filters them with a cross-encoder relevance gate, and asks DeepSeek V3.1 for an answer that must be grounded in those passages, or refuses.

Stack: FastAPI, FAISS, BAAI/bge-m3 embeddings, BAAI/bge-reranker-v2-m3, DeepSeek V3.1 (Huawei ModelArts MaaS), PyTorch, Gradio.

> The repository was first named "Flask-Forsa", but it has always been a FastAPI app.

## Architecture

```
POST /api/chat/   { "equipe": ..., "question": { category: { id: text } } }
        │
        ▼
 build_prompts(questions)                      api/chat.py, one batch per request
        │
        ├─ query analysis                      services/query_service.py
        │    normalize, key terms, document family
        ├─ semantic search                     bge-m3 (fp16, GPU) → FAISS IndexFlatIP, 945 chunks
        │    one encode call + one FAISS search for the whole batch
        ├─ lexical search                      keyword scorer over cached lowercase chunk text
        ├─ hybrid merge                        min-max normalized, 0.65 semantic + 0.35 lexical
        ├─ relevance gate                      bge-reranker-v2-m3 (fp16) scores the top 10 candidates,
        │                                      keeps chunks with sigmoid score ≥ 0.05, at most 6;
        │                                      same-language chunks win near-ties
        └─ scope filter + prompt               utils/validation_utils.py, prompts.py
        │
        ▼
 answer_question(...)                          concurrent, one per question
        ├─ no chunk passed the gate → "Information non disponible dans les documents fournis."
        │                              (no LLM call)
        └─ DeepSeek V3.1, JSON mode, pooled aiohttp session → grounding check (logged)
```

Indexing is offline: `preprocess.py` asks DeepSeek to rewrite each PDF/DOCX in `data/` as Markdown chunks in `gen/` (one file per chunk), and `rebuild_index.py --keep-chunks` (or `POST /api/embeddings/gen`) embeds them into `db/faiss_index.bin`, replacing the previous index.

## Performance and retrieval quality

All numbers come from `eval/run_eval.py`. It runs everything before the LLM call through the same `build_prompts` function the endpoint uses, on an RTX 4070 Laptop GPU, with no network calls. The eval set is `eval/questions.json`: questions written from the indexed corpus (35 French, 15 Arabic), each naming the documents that contain the answer and the keywords a correct answer must include, plus off-topic questions that should be refused.

| | Before | After, no reranker | After, reranker (20 candidates, no threshold) |
|---|---|---|---|
| Index | 3,092 vectors (each chunk 3×) | 945 unique chunks | 945 unique chunks |
| Startup (models + index load) | 32.4 s (both models) | 18.8 s | 26.2 s, reranker loads on first query |
| Peak GPU memory | 4,342 MB | 1,093 MB | 2,404 MB |
| Retrieval per question, mean | 86 ms | 53 ms | 231 ms |
| … embedding / lexical / rerank | 47 / 38 / – ms | 46 / 4 / – ms | 65 / 5 / 159 ms |
| 55-question request, retrieval wall time | 5.74 s (thread per question) | 0.50 s (batched) | 9.72 s (batched) |
| Source document ranked 1st (hit@1) | 0.92 | 0.98 | 0.96 |
| hit@3 | 1.00 | 1.00 | 1.00 |
| MRR | 0.957 | 0.990 | 0.980 |
| hit@1 French / Arabic | 0.91 / 0.93 | 0.97 / 1.00 | 0.94 / 1.00 |
| Answer keywords present in the LLM context | 0.99 | 1.00 | 1.00 |
| Chunks sent to the LLM | 13.7 | 18.8 | 18.9 |
| Off-topic questions refused | 0 / 5 | 0 / 5 | 0 / 5 |

What changed and why:

- **The index held every chunk three times.** `POST /api/embeddings/gen` appended to the index instead of replacing it, and it had been run three times. The index builder now rebuilds from scratch and drops duplicate texts. 12.7 MB → 3.9 MB.
- **The reranker loaded at startup even when disabled**, holding 2.2 GB of GPU memory. It now loads on first use, and both models run in fp16.
- **Lexical search lowercased the whole corpus on every query.** The lowercase text is now cached: 38 ms → 4 ms.
- **A multi-question request ran one thread per question**, and those threads contended for the GPU and the GIL: 55 questions took 5.7 s, slower than running them back to back. Retrieval now runs as one batch, taking 0.5 s without the reranker.
- **The scope filter only looked at file names.** Every file name used to carry a broken "...Idoom Medium Business..." folder prefix, so the filter never removed anything. With clean names it started dropping the Weekend Boost and Gamers sheets, so it now also checks the chunk text.
- **Most documents exist as a French and an Arabic twin with near-identical scores.** Chunks in the question's language now win ties; the prompt requires answering in that language anyway.

### Reranking decision

On this set the reranker does not improve ranking: hybrid retrieval already puts the right document first 98% of the time. What it adds is an absolute relevance score. Hybrid scores are min-max normalized per query, so the best chunk always scores 1.0 and no threshold on them can tell an answerable question from an off-topic one. Sweeping the reranker threshold on the 20-candidate run, cutting to the 6 sources the prompt receives:

| Threshold | hit@1 | Source in context | In-scope refused | Off-topic refused | Chunks sent |
|---|---|---|---|---|---|
| hybrid only, top 6 | 0.98 | 1.00 | 0 | 0 / 5 | 6.0 |
| 0 | 0.96 | 1.00 | 0 | 0 / 5 | 6.0 |
| 0.01 | 0.96 | 1.00 | 0 | 5 / 5 | 5.7 |
| **0.05 (chosen)** | 0.96 | 1.00 | 0 | 5 / 5 | 5.3 |
| 0.1 | 0.96 | 1.00 | 0 | 5 / 5 | 5.1 |
| 0.3 | 0.96 | 1.00 | 0 | 5 / 5 | 4.1 |
| 0.5 | 0.94 | 0.98 | 1 | 5 / 5 | 3.6 |

The highest score any off-topic question reached was 0.0084. Reranking is therefore **on**, with `RERANK_MIN_SCORE = 0.05`, at most 6 chunks per question, over the top 10 hybrid candidates (the right document was always within the first two). Off-topic questions are refused without an LLM call, and the prompt carries about 5 chunks instead of 19. The cost measured with 20 candidates was 159 ms of GPU time per question (scoring 10 should take about half; not measured), which is small next to an LLM call.

The LLM answer fixer (a second LLM pass that rewrote each answer) was removed. It was disabled, it would have doubled LLM time and cost per question, and its rules duplicate the system prompt.

### Not measured in this pass

- **LLM latency and answer accuracy.** Both need a working DeepSeek key; the keys that were in the repo are rejected by DeepSeek and Gemini, and this pass made no paid API calls. `answerKeywords` in the eval set is there so answer accuracy can be scored once one is available. The old README claimed 45–65 s per question; that could not be reproduced. The December 2025 run logs show 1.1–1.6 s of retrieval and about 3 s per LLM call, with a 59-question request finishing in 34.6 s.
- **The shipped configuration end to end.** The table's reranker column uses 20 candidates and no threshold; the threshold rows are replayed from that run. The final settings (10 candidates, threshold 0.05) and five harder near-domain off-topic questions (Djezzy, Mobilis, Algérie Poste) are in the repo, but their run did not complete: the machine ran out of memory.
- Raw results are committed only for the "before" run (`eval/results/baseline.json`). The two "after" columns were measured on 55 questions (50 in scope, 5 off-topic) with the code and index of commit `30dcd53` and the `--set` overrides below, before that commit's five extra off-topic questions existed; their JSON was not kept. Rerun them with the commands below.

### Reproduce

```bash
python eval/run_eval.py --label noRerank --batch --set USE_RERANKER=false
```

```bash
python eval/run_eval.py --label rerankSweep --batch --set RERANK_CANDIDATES=20 --set RERANK_MIN_SCORE=0 --set MAX_SOURCES=20
```

```bash
python eval/threshold_sweep.py eval/results/rerankSweep.json
```

```bash
python eval/run_eval.py --label final --batch
```

Each run writes `eval/results/<label>.json` with a summary and per-question rows (rank of the source document, context documents and scores, stage timings). It needs about 2.5 GB of GPU memory with the reranker and makes no API calls.

## Setup

Python 3.10+ and a CUDA GPU are recommended; on CPU both models run in fp32 and are much slower.

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env.local` and set `DEEPSEEK_TOKEN`. It is only needed for answering and for `preprocess.py`; retrieval and the eval run without it.

The committed index in `db/` covers the current corpus. To rebuild it from your own documents:

```bash
python preprocess.py
```

```bash
python rebuild_index.py --keep-chunks
```

Run the API (docs at http://localhost:8000/docs):

```bash
python main.py
```

Optional Gradio UI on http://localhost:7860, talking to the API:

```bash
python gradio_app.py
```

## API

`POST /api/chat/` takes any number of questions grouped by category and answers them in one call:

```json
{
  "equipe": "GhostTruth",
  "question": {
    "Gamers": { "q1": "Quel débit maximum atteint l'offre Idoom Fibre Gamers ?" },
    "EKOTEB": { "q2": "كيف أشترك في تطبيق EKOTEB؟" }
  }
}
```

```json
{
  "equipe": "GhostTruth",
  "reponses": {
    "Gamers": { "q1": "..." },
    "EKOTEB": { "q2": "..." }
  }
}
```

Other routes: `POST /api/embeddings/gen` rebuilds the index from `gen/`, `POST /api/embeddings/preprocess` runs the DeepSeek chunker, `POST /api/embeddings/add` appends documents, and `POST /api/cache/reload` reloads the index from disk.

## Configuration

`config.py`, with the values that matter most:

| Setting | Value | Meaning |
|---|---|---|
| `TOP_K_SEMANTIC`, `TOP_K_LEXICAL` | 20, 20 | candidates from each retriever |
| `HYBRID_W_SEM`, `HYBRID_W_LEX` | 0.65, 0.35 | fusion weights |
| `USE_RERANKER` | True | relevance gate on/off |
| `RERANK_CANDIDATES` | 10 | hybrid candidates scored per question |
| `RERANK_MIN_SCORE` | 0.05 | sigmoid score a chunk needs to reach the LLM |
| `MAX_SOURCES` | 6 | chunks per prompt when reranking |
| `SAME_LANGUAGE_BONUS` | 0.05 | ranking bonus for chunks in the question's language |
| `MODELS_FP16` | True | fp16 models on CUDA |
| `LLM_MAX_CONNECTIONS` | 20 | pooled connections to the LLM API |

## Project layout

```
api/chat.py                 /api/chat: batched retrieval, refusals, LLM calls
api/embeddings.py           index rebuild, preprocessing, document upload
services/rag_service.py     hybrid retrieval, reranker gate, language tiebreak
services/embedding_service.py, reranker_service.py, vector_store.py, query_service.py
services/index_builder.py   gen/*.md → FAISS index
services/llm_service.py     DeepSeek client (sync + pooled async)
utils/validation_utils.py   scope filter and answer grounding checks
prompts.py                  system / user prompts
preprocess.py               DeepSeek Markdown chunker for data/
eval/                       questions, eval runner, threshold sweep, results
```

## License

© 2025. All rights reserved. Unauthorized use, reproduction, modification, or distribution of this project or its source code is prohibited.
