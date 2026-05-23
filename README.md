# ⚖️ Legal Case Prediction using RAG (Retrieval-Augmented Generation)

> **GAI48** — A production-grade AI system that retrieves relevant past legal judgments and predicts case outcomes with cited precedents, using structured legal reasoning and citation chains.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-red.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.9+-orange.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What Does This Project Do?

Imagine you are a lawyer. You have a new case in front of you. Normally, you would spend **hours or days** searching through thousands of past court judgments to find similar cases and understand how courts ruled on them. 

This project **automates that process using AI**:

1. You give the system a description of your case (the facts)
2. The system **searches through thousands of real court judgments** (ECHR, Indian Kanoon, SCOTUS)
3. It finds the **most relevant past cases** using vector similarity search (FAISS)
4. It predicts the **likely outcome** of your case
5. It shows you **which past cases support the prediction**, with citations

This is called **RAG — Retrieval-Augmented Generation**. The AI doesn't just guess — it **retrieves real evidence first, then reasons on top of it**.

---

## 🏗️ System Architecture

```
User Input (Case Facts)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                   FastAPI Backend                    │
│                  (api/main.py)                       │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              RAG Pipeline (src/retrieval)            │
│                                                      │
│  1. Embed query → vector (using sentence-transformers│
│  2. FAISS similarity search → top-K cases           │
│  3. NetworkX citation graph → related cases         │
│  4. LangChain RetrievalQA → structured answer       │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│           Prediction Engine (src/prediction)         │
│                                                      │
│  - Confidence-calibrated outcome prediction          │
│  - Citation chain reasoning                          │
│  - MLflow experiment tracking                        │
└─────────────────────────────────────────────────────┘
        │
        ▼
  JSON Response with:
  - Predicted Outcome (violation / no violation)
  - Confidence Score
  - Top Cited Cases (with article references)
  - Citation Graph path
```

---

## 📁 Project Structure

```
legal-rag-project/
│
├── 📂 data/                         # Legal datasets (not pushed to git - too large)
│   ├── echr_cases.json              # European Court of Human Rights judgments
│   ├── scotus_cases.json            # US Supreme Court decisions
│   └── kanoon_cases.json            # Indian legal cases from Kaggle
│
├── 📂 src/
│   ├── 📂 ingestion/                # Step 1: Load and chunk legal documents
│   │   ├── __init__.py
│   │   ├── document_loader.py       # Loads JSON/CSV datasets
│   │   └── chunker.py              # Section-aware splitting of legal docs
│   │
│   ├── 📂 retrieval/                # Step 2: Store and search embeddings
│   │   ├── __init__.py
│   │   ├── embedder.py             # Converts text → vectors
│   │   ├── faiss_store.py          # FAISS vector database operations
│   │   └── citation_graph.py       # NetworkX citation relationship graph
│   │
│   ├── 📂 prediction/               # Step 3: LangChain QA + outcome prediction
│   │   ├── __init__.py
│   │   ├── rag_chain.py            # LangChain RetrievalQA pipeline
│   │   └── predictor.py            # Final outcome classifier + confidence
│   │
│   └── 📂 api/                      # Step 4: FastAPI REST endpoints
│       ├── __init__.py
│       └── main.py                 # API routes and request/response models
│
├── 📂 notebooks/
│   └── exploration.ipynb           # EDA, dataset exploration, testing
│
├── 📂 tests/
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── 📂 .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
│
├── 📄 requirements.txt             # All Python dependencies
├── 📄 Dockerfile                   # Container definition
├── 📄 docker-compose.yml           # Multi-service setup (API + MLflow)
├── 📄 mlflow_config.py             # MLflow experiment configuration
├── 📄 .env.example                 # Environment variables template
└── 📄 README.md                    # This file
```

---

## 🧠 Key Concepts Explained (For Interviews)

### What is RAG?
**RAG = Retrieval-Augmented Generation**. Instead of asking an LLM to answer from memory (which can hallucinate), you first **retrieve real documents** from a database, then pass them to the LLM as context. The LLM can only answer using what you retrieved. This makes it **factual and cited**.

### What is FAISS?
**FAISS (Facebook AI Similarity Search)** is a library that stores millions of text embeddings (vectors) and finds the most similar ones to a query in milliseconds. Think of it as Google Search, but for meaning, not keywords.

### What is a Citation Graph?
Legal cases reference each other. If Case A cited Case B, and Case B cited Case C, there's a graph of relationships. Using **NetworkX**, we build this graph so we can find not just directly similar cases, but also cases that are **indirectly related** through citation chains.

### What is LangChain RetrievalQA?
LangChain provides a ready-made pipeline: given a question, it retrieves relevant documents from a vector store, formats them as context, and sends them to an LLM (Anthropic/OpenAI) to generate a structured answer.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/abishekgamoth/legal-rag-project.git
cd legal-rag-project
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# OR
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 5. Run the ingestion pipeline (build the FAISS index)
```bash
python -m src.ingestion.document_loader
python -m src.retrieval.faiss_store --build
```

### 6. Start the API
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 7. Test the API
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"case_facts": "The applicant was detained without trial for 6 months with no legal representation provided."}'
```

---

## 🐳 Run with Docker

```bash
docker-compose up --build
```

This starts:
- **Port 8000**: FastAPI (your main app)
- **Port 5000**: MLflow tracking UI

---

## 📊 MLflow Experiment Tracking

Every prediction is logged to MLflow:
- `retrieval_precision`: How relevant were the retrieved cases?
- `prediction_confidence`: How confident is the model?
- `citation_relevance`: Are citations actually related to the query?
- `response_latency`: How fast did the system respond?

View the dashboard:
```bash
mlflow ui --port 5000
# Open: http://localhost:5000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Main prediction endpoint |
| `GET` | `/cases/search` | Search similar cases |
| `GET` | `/citation-graph/{case_id}` | Get citation graph for a case |
| `GET` | `/health` | Health check |

### Example Request
```json
POST /predict
{
  "case_facts": "The applicant was detained without trial for 6 months with no legal representation."
}
```

### Example Response
```json
{
  "predicted_outcome": "VIOLATION",
  "confidence": 0.87,
  "article_violated": "Article 5 - Right to Liberty",
  "cited_cases": [
    {
      "case_id": "ECHR-2019-001",
      "title": "Smith v. United Kingdom",
      "relevance_score": 0.92,
      "outcome": "VIOLATION",
      "summary": "Detention without charge for 7 months violated Article 5"
    }
  ],
  "reasoning": "Based on 3 similar cases, courts consistently ruled violations when detention exceeded 3 months without trial...",
  "citation_chain": ["ECHR-2019-001", "ECHR-2017-045", "ECHR-2015-022"]
}
```

---

## 📈 Datasets Used

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| ECHR Cases | [Kaggle](https://kaggle.com/datasets/echr) | ~11,000 cases | European Court of Human Rights judgments |
| Indian Kanoon | [Kaggle](https://kaggle.com/datasets/indian-kanoon) | ~5,000 cases | Indian Supreme Court decisions |
| SCOTUS | [Kaggle](https://kaggle.com/datasets/scotus) | ~8,000 cases | US Supreme Court decisions |

---

## 🛠️ Tech Stack

| Component | Technology | Why? |
|-----------|-----------|------|
| Embeddings | `sentence-transformers` | Converts legal text to vectors |
| Vector Search | `FAISS` | Fast similarity search |
| Citation Graph | `NetworkX` | Graph-based case relationships |
| LLM Pipeline | `LangChain` | RetrievalQA chain |
| LLM | `Anthropic Claude / OpenAI GPT` | Final answer generation |
| API | `FastAPI` | High-performance REST API |
| Tracking | `MLflow` | Experiment and metric logging |
| CI/CD | `GitHub Actions` | Automated testing + deployment |
| Container | `Docker` | Reproducible environment |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 👤 Author

**Amgoth Abhishek**  
- 🔗 [LinkedIn](https://linkedin.com/in/abhishek-amgoth)
- 🐙 [GitHub](https://github.com/abishekgamoth)
- 📧 abhishek.am23@iiits.in

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
