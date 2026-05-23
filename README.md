# ⚖️ Casechain-AI

AI-powered Legal Case Prediction system using RAG (Retrieval-Augmented Generation).

The system retrieves relevant past court judgments using FAISS semantic search and citation graphs, then predicts legal outcomes with cited precedents using LLMs.

---

## 🚀 Features

- 🔍 Semantic legal case search with FAISS
- ⚖️ Legal outcome prediction
- 🧠 RAG pipeline with LangChain
- 🔗 Citation graph using NetworkX
- ⚡ FastAPI backend
- 🐳 Docker support
- 📊 MLflow experiment tracking
- ✅ Unit testing & CI/CD

---

## 🛠️ Tech Stack

- Python
- FastAPI
- LangChain
- FAISS
- NetworkX
- Sentence Transformers
- Docker
- MLflow

---

## 📁 Project Structure

```bash
src/
 ├── api/
 ├── ingestion/
 ├── retrieval/
 └── prediction/
```

---

## ⚡ Run Locally

### Clone Repository

```bash
git clone https://github.com/abishekamgoth/Casechain-ai.git
cd Casechain-ai
```

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run FastAPI Server

```bash
uvicorn src.api.main:app --reload
```

Open:

```bash
http://127.0.0.1:8000/docs
```

---

## 🐳 Run with Docker

```bash
docker-compose up --build
```

---

## 🧪 Run Tests

```bash
pytest
```

---

## 📌 Example API Endpoint

```http
POST /predict
```

Request:

```json
{
  "case_facts": "The applicant was detained without trial for 6 months."
}
```

---

## 👨‍💻 Author

Amgoth Abhishek

- GitHub: https://github.com/abishekamgoth

---

## 📄 License

MIT License
