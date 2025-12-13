# 🤖 Flask-Forsa RAG System

**A high-performance, production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, featuring hybrid search, semantic reranking, and bilingual support (French/Arabic).**

Built with **FastAPI + Python + DeepSeek/Gemini + FAISS + BGE-M3**.

---

# 🚀 Tech Stack

## **Backend Framework**

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

## **AI & ML Stack**

![DeepSeek](https://img.shields.io/badge/DeepSeek_V3.1-000000?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

## **Vector & Search**

![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![BGE-M3](https://img.shields.io/badge/BGE--M3_Embeddings-FF6B6B?style=for-the-badge)
![BGE Reranker](https://img.shields.io/badge/BGE_Reranker_V2-4ECDC4?style=for-the-badge)

## **Data Processing**

![PyMuPDF](https://img.shields.io/badge/PyMuPDF-FF0000?style=for-the-badge)
![python-docx](https://img.shields.io/badge/python--docx-2B579A?style=for-the-badge)

## **UI/Interface**

![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge)

---

# ⚡ Overview

Flask-Forsa is a **production-grade RAG system** designed for high-accuracy question answering with strict source attribution.

## Core Capabilities

* **Hybrid Retrieval** — Semantic + Lexical search fusion with MinMax normalization
* **Smart Reranking** — BGE-reranker-v2-m3 cross-encoder for precision
* **Bilingual Support** — Native French & Arabic processing
* **Parallel Processing** — Handle 50+ questions concurrently via ThreadPoolExecutor
* **JSON Mode** — Structured, reliable LLM outputs with validation
* **Answer Validation** — Grounding checks & hallucination detection
* **Source Attribution** — Evidence-based answers with score-based citations
* **Document Processing** — PDF, DOCX, TXT support with Gemini semantic chunking

Everything runs **locally** with **full control** over your data pipeline.

---

# 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              FastAPI REST API Endpoint                   │
│      (Structured Chat + asyncio.gather Parallelism)      │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐              ┌──────▼──────┐
│ Query Service  │              │ RAG Service │
│ (Enhancement)  │              │  (Hybrid)   │
│ • Normalize    │              │ • Semantic  │
│ • Extract      │              │ • Lexical   │
│ • Variations   │              │ • Merge     │
└───────┬────────┘              └──────┬──────┘
        │                               │
        │         ┌─────────────────────┤
        │         │                     │
┌───────▼─────────▼───┐      ┌─────────▼──────────┐
│  Embedding Service  │      │  Vector Store      │
│    (BGE-M3 1024d)   │      │  (FAISS Index)     │
│ • GPU Accelerated   │      │ • Flat Index       │
│ • Batch Processing  │      │ • Cosine Sim       │
└─────────────────────┘      └─────────┬──────────┘
                                       │
                             ┌─────────▼──────────┐
                             │ Reranker Service   │
                             │ (BGE Reranker V2)  │
                             │ • Cross-Encoder    │
                             │ • Score Override   │
                             └─────────┬──────────┘
                                       │
                             ┌─────────▼──────────┐
                             │  Validation Utils  │
                             │ • Scope Filter     │
                             │ • Grounding Check  │
                             └─────────┬──────────┘
                                       │
                             ┌─────────▼──────────┐
                             │   LLM Service      │
                             │ • DeepSeek V3.1    │
                             │ • JSON Mode        │
                             │ • Retry Logic      │
                             └────────────────────┘
```

---

# 🧠 Core Algorithms

| Feature             | Algorithm                      | Description                                    |
| ------------------- | ------------------------------ | ---------------------------------------------- |
| Hybrid Search       | Weighted Semantic + Lexical    | MinMax normalization with 0.65/0.35 weights    |
| Reranking           | BGE Cross-Encoder Scoring      | Neural reranking with score replacement        |
| Query Enhancement   | Multi-pass Normalization       | Key term extraction + variations               |
| Scope Filtering     | Entity + Family Detection      | Pre-filters chunks by question context         |
| Answer Validation   | Grounding + Citation Analysis  | Checks answer support in sources               |
| Deduplication       | Hash-based Chunk Keys          | SHA1 content hashing prevents duplicates       |
| Document Chunking   | Gemini Semantic Splitting      | Context-aware chunk boundaries                 |
| Parallel Execution  | asyncio.gather + Pure Async    | Unlimited concurrent async operations          |

---

# 💽 Installation Guide

## 1. Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for embeddings)

## 2. Clone & Setup

```bash
# Clone repository
git clone <repository-url>
cd flask-forsa

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Configure Environment

Create `.env.local`:

```env
# LLM API Keys
DEEPSEEK_TOKEN=your_deepseek_api_token
DEEPSEEK_API_URL=https://api.modelarts-maas.com/v2/chat/completions
GEMINI_API_KEY=your_gemini_api_key
```

## 4. Prepare Your Data

```bash
# Add documents to data/ folder
mkdir -p data
# Copy your PDFs, DOCX files here

# Run preprocessing with Gemini
python preprocess.py
```

## 5. Build Vector Index

```bash
# Index your preprocessed chunks
python build_index.py

# Rebuild the index from scratch if needed
python rebuild_index.py
```

## 6. Launch Server

```bash
# Start FastAPI server
python main.py

# Server runs at: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## 7. (Optional) Launch Gradio UI

```bash
# In a new terminal
python gradio_app.py

# UI runs at: http://localhost:7860
```

---

# 📚 API Usage

## Structured Chat Endpoint

**Endpoint:** `POST /api/chat/`

### Request Schema

```json
{
  "equipe": "GhostTruth",
  "question": {
    "category_1": {
      "q1": "Quel est le dossier administratif requis pour Idoom?",
      "q2": "Quels sont les avantages de l'offre Idoom Fibre?"
    },
    "category_2": {
      "q1": "Comment souscrire à Idoom 4G LTE?"
    }
  },
  "include_sources": false
}
```

### Response Schema

```json
{
  "equipe": "GhostTruth",
  "reponses": {
    "category_1": {
      "q1": "Le dossier administratif comprend...",
      "q2": "Les avantages de l'offre Idoom Fibre incluent..."
    },
    "category_2": {
      "q1": "Pour souscrire à Idoom 4G LTE, vous devez..."
    }
  },
  "sources": null
}
```

### With Source Attribution

Set `"include_sources": true`:

```json
{
  "sources": {
    "category_1": {
      "q1": [
        {
          "file": "guide_idoom_2024.pdf",
          "score": 0.8543,
          "evidence": "Le dossier administratif requis comprend une copie de la CIN..."
        },
        {
          "file": "procedures_algerie_telecom.pdf",
          "score": 0.7821,
          "evidence": "Documents nécessaires : justificatif de domicile, CIN..."
        }
      ]
    }
  }
}
```

---

# ⚙️ Configuration Reference

Edit `config.py` to customize behavior:

## Retrieval Settings

```python
TOP_K_SEMANTIC = 20          # Semantic search candidates
TOP_K_LEXICAL = 20           # Lexical search candidates
HYBRID_W_SEM = 0.65          # Semantic weight (0-1)
HYBRID_W_LEX = 0.35          # Lexical weight (0-1)
SIMILARITY_THRESHOLD = 0     # Minimum hybrid score (0 = disabled)
```

## Reranking

```python
USE_RERANKER = False                           # Enable/disable reranking
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"    # Cross-encoder model
```

## Answer Processing

```python
ENABLE_ANSWER_FIXER = False      # Strict RAG post-processing (not used)
ANSWER_FIXER_TEMPERATURE = 0.1   # LLM temperature for fixes
```

## Vector Store

```python
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # Embedding model
EMBEDDING_DIMENSION = 1024             # Embedding dimensions
INDEX_TYPE = "Flat"                    # FAISS index type
FAISS_INDEX_PATH = "db/faiss_index.bin"
```

## Source Limits

```python
MIN_SOURCES = 1   # Minimum sources to return
MAX_SOURCES = 6   # Maximum sources to return
```

## LLM Settings

```python
MODEL_NAME = "deepseek-v3.1"          # DeepSeek model
GEMINI_MODEL = "gemini-2.5-flash"     # Gemini model for chunking
ENABLE_LLM_WARMUP = False             # Warmup on startup
```

---

# 📊 Performance Metrics

| Metric                    | Performance          |
| ------------------------- | -------------------- |
| Parallel Questions        | Up to 50 concurrent  |
| Avg Latency (uncached)    | ~45-65s per question |
| Retrieval Speed           | ~50s (hybrid + rerank) |
| Reranking Speed           | ~40-60s (BGE reranker) |
| LLM Generation            | 1-3s (DeepSeek)      |
| Embedding Batch (GPU)     | ~100 docs/s          |
| Index Search (FAISS)      | <1s                  |

**Note:** Current configuration has reranking disabled (`USE_RERANKER = False`) and threshold at 0.

There is an in-progress latency-tuning effort tracked in `PERFORMANCE_OPTIMIZATIONS.md` (and a `QUICKSTART_PERFORMANCE.md` companion doc) — not yet merged into this README, treat those as work-in-progress notes rather than final numbers.

---

# 🔧 Advanced Features

## Query Enhancement

Automatically processes queries via `query_service`:

- **Normalization**: Lowercase, diacritics removal
- **Key Terms**: Extracts important words for lexical search
- **Variations**: Generates alternative phrasings
- **Type Detection**: Identifies question category

## Scope-Based Filtering

Intelligently filters chunks via `validation_utils`:

- **Entity Detection**: Companies, products, services
- **Document Families**: Categorizes by topic
- **Question Type**: Matches chunk relevance

## Answer Validation

Multi-layer validation in `validation_utils`:

- **Grounding Check**: Verifies answer in sources
- **Hallucination Detection**: Flags unsupported claims
- **Availability Check**: Detects "no info" responses
- **Citation Accuracy**: Validates source references

## Document Processing

Advanced chunking with Gemini in `preprocess.py`:

- **Semantic Boundaries**: Context-aware splits
- **Metadata Extraction**: File, page, section info
- **Multi-format**: PDF, DOCX, TXT support via `document_processor`

---

# 📁 Project Structure

```
flask-forsa/
├── api/
│   ├── chat.py              # Main structured chat endpoint
│   ├── embeddings.py        # Embedding utilities
│   └── cache.py             # Cache management endpoints
├── services/
│   ├── rag_service.py       # Hybrid retrieval engine
│   ├── llm_service.py       # LLM integration (DeepSeek)
│   ├── embedding_service.py # BGE-M3 embeddings
│   ├── vector_store.py      # FAISS vector operations
│   ├── reranker_service.py  # BGE reranker (optional)
│   ├── query_service.py     # Query enhancement
│   └── document_processor.py # PDF/DOCX extraction
├── utils/
│   ├── validation_utils.py  # Answer validation & scope filtering
│   ├── answer_fixer.py      # Post-processing (not used)
│   └── logging_utils.py     # Performance tracking decorator
├── data/                    # Document storage
├── db/                      # FAISS index + metadata
├── schemas.py               # Pydantic models
├── config.py                # Configuration
├── prompts.py               # System prompts
├── preprocess.py            # Gemini-based document chunking
├── build_index.py           # FAISS index builder
├── main.py                  # FastAPI app
├── gradio_app.py            # Gradio UI
├── test_preprocess.py       # Gemini test suite
└── README.md                # This file
```

---

# 🧪 Testing

## Test Gemini Setup

```bash
python test_preprocess.py
```

## Other Test Suites

The repo also includes targeted tests for chunking, answer fixing, strict-RAG rules, and the API surface:

```bash
python test_ai_chunks.py
python test_answer_fixer.py
python test_strict_rag_rules.py
python test_api.py
python test_structured.py
```

## Test API with cURL

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "equipe": "TestTeam",
    "question": {
      "test": {
        "q1": "Comment souscrire à Idoom?"
      }
    },
    "include_sources": true
  }'
```

## Run with Gradio UI

```bash
# Terminal 1: Start API
python main.py

# Terminal 2: Start Gradio
python gradio_app.py
```

Visit `http://localhost:7860` for the web interface.

---

# 📈 Monitoring & Debugging

## Enable Debug Logs

In `config.py`:

```python
DEBUG_LOG_RETRIEVAL = True
```

## Log Output Includes

- **Retrieval Transparency**: Hybrid scores, semantic/lexical breakdown
- **Query Enhancement**: Normalization steps, key terms, variations
- **Validation Results**: Scope filtering, grounding checks
- **Performance Metrics**: Timing for each pipeline stage via `@log_execution_time`
- **Token Usage**: DeepSeek input/output tokens

## Current Logging

Logs are configured via `logger_config.py` and output to console with INFO level.

---

# 🤝 Contributing

This is a **private project**. Unauthorized use, reproduction, or distribution is prohibited.

For authorized contributors:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

# 📄 License

```
© 2025. All rights reserved.
Unauthorized use, reproduction, modification, or distribution
of this project or its source code is strictly prohibited.
```

**⚠️ IMPORTANT:** This project is **private** and **fully restricted**.

---

# 🙏 Acknowledgments

**AI & ML Models:**
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — Multilingual embeddings (1024d)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — Cross-encoder reranking
- [DeepSeek V3.1](https://www.deepseek.com/) — Advanced language model with JSON mode
- [Google Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/) — Semantic document chunking

**Infrastructure:**
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI vector similarity search
- [FastAPI](https://fastapi.tiangolo.com/) — Modern async Python web framework
- [Gradio](https://gradio.app/) — ML web interfaces
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF processing and text extraction
- [Transformers](https://huggingface.co/docs/transformers/) — HuggingFace ML model library

---

**Built with ❤️ for production-grade RAG applications**

*Powered by hybrid search, semantic reranking, and bilingual intelligence.*
