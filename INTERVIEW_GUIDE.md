# 🎤 How to Explain This Project — Interview Guide

> This guide is **specifically for you, Abhishek**, to confidently explain the Legal RAG project in interviews, hackathons, or to professors.

---

## 📌 30-Second Elevator Pitch

> "I built a Retrieval-Augmented Generation system for legal case prediction. The system retrieves relevant past court judgments using semantic vector search with FAISS, builds a citation graph with NetworkX to capture how cases reference each other, then uses an LLM through LangChain to predict whether a new case will result in a rights violation — with actual cited precedents in the answer. The whole thing is served via a FastAPI REST API, tracked with MLflow, and containerized with Docker."

---

## 🔑 Key Concept Questions & Answers

### Q: What is RAG and why did you use it?

**Answer:**
RAG stands for Retrieval-Augmented Generation. The problem with asking an LLM a legal question directly is that it might hallucinate case names or outcomes. In law, accuracy is critical — you can't cite a case that doesn't exist.

RAG solves this by splitting the problem into two steps:
1. **Retrieve**: First, search a database of real court cases for similar ones.
2. **Generate**: Then give those real cases to the LLM as context and ask it to reason on top of them.

The LLM can only use what we retrieved. So the citations in the answer are always real cases from our database.

---

### Q: What is FAISS and why not just use a regular database?

**Answer:**
FAISS is Facebook's library for similarity search in high-dimensional vector spaces.

A regular database (like SQL) searches by exact keywords. If you search "arrested without charge", it won't find a case that says "detained without formal accusation" even though they mean the same thing.

FAISS works differently. We convert every legal case into a vector — a list of 384 numbers — using a sentence embedding model. These numbers capture the semantic meaning of the text. Similar meanings → similar vectors → close together in the 384-dimensional space.

When a user submits case facts, we embed their query into the same space and ask FAISS: "which stored vectors are closest?" It returns cases with similar meaning, not just similar words.

---

### Q: What is the citation graph and why is it important?

**Answer:**
Legal reasoning isn't just about semantic similarity. Courts follow precedents — older cases that established the legal principle. These relationships are captured in citations.

I built a directed graph using NetworkX where:
- Each node = one legal case
- Each directed edge = "Case A cited Case B" (A depends on B)

This lets me find cases that are related through citation chains, even if they use completely different language. A landmark case from 1978 might be the original precedent for a 2023 case — FAISS might miss this because the writing styles are so different. The graph catches it.

I also ran PageRank on the graph — the same algorithm Google originally used — to identify the most "important" landmark cases (those cited by many other important cases).

---

### Q: Explain the pipeline step by step.

**Answer:**
1. **Ingestion**: Load legal case documents from ECHR, SCOTUS, and Indian Kanoon datasets. Clean and normalize them into a unified format.

2. **Chunking**: Legal cases can be 10,000+ words, but embedding models handle ~400 words max. So I split each case into overlapping chunks, section-aware: facts section, arguments section, judgment section. Keeping sections intact preserves context.

3. **Embedding**: Use `sentence-transformers` to convert every chunk into a 384-dimensional vector. Store all vectors in a FAISS index on disk.

4. **Citation Graph**: Build a NetworkX directed graph where edges represent citation relationships between cases.

5. **Prediction**: When a user submits case facts → embed the query → FAISS returns top-5 similar chunks → Citation graph adds related cases → LangChain formats context into a legal prompt → Claude/GPT reads the context and predicts outcome with citations.

6. **API**: Everything is exposed via FastAPI REST endpoints. POST /predict is the main one.

7. **Tracking**: Every prediction logs metrics (confidence, latency, outcome) to MLflow for monitoring and analysis.

---

### Q: What is LangChain used for here?

**Answer:**
LangChain provides the "chain" that connects retrieval to generation. Specifically, I used:
- `PromptTemplate`: A carefully designed legal prompt that tells the LLM to use ONLY the retrieved cases and to output in a structured format (OUTCOME: / CONFIDENCE: / REASONING:).
- `ChatAnthropic / ChatOpenAI`: LangChain wrappers for calling the LLM APIs that handle retries, formatting, and streaming.

The custom legal prompt is important — generic prompts give generic answers. My prompt instructs the LLM to cite specific cases by name and explain which legal articles apply.

---

### Q: What is MLflow tracking?

**Answer:**
MLflow is an open-source platform for tracking machine learning experiments. Every time someone uses my /predict endpoint, I log:
- `confidence`: How certain was the prediction (0.0-1.0)?
- `retrieval_time_ms`: How long did FAISS search take?
- `total_time_ms`: Total latency of the request
- `num_cited_cases`: How many precedents were found?
- `is_violation`: Did we predict a violation? (useful for analyzing patterns)

This creates a history of all predictions I can analyze. If accuracy drops or latency increases, MLflow lets me pinpoint when it started and what changed.

---

### Q: Why Docker?

**Answer:**
Docker packages the application and all its dependencies into a container — a lightweight, isolated environment. This solves the classic "works on my machine" problem.

Without Docker: "I need Python 3.10, install these 20 packages, set these environment variables..."
With Docker: One command — `docker-compose up` — starts everything identically on any machine.

I used docker-compose to run two services together: the FastAPI app on port 8000, and the MLflow dashboard on port 5000.

---

### Q: What datasets did you use?

**Answer:**
1. **ECHR Dataset** (European Court of Human Rights): ~11,000 real court judgments from the Strasbourg court. Each case documents whether a European country violated someone's rights under the European Convention. Outcomes are "violation" or "no violation" of specific Articles.

2. **Indian Kanoon Dataset** (Kaggle): Indian Supreme Court cases. Useful for including cases from a different legal system.

3. **SCOTUS Dataset**: US Supreme Court decisions, which have well-structured data including the legal issue area and decision direction.

---

### Q: What are the limitations of your system?

**Answer** (showing maturity and self-awareness):
1. **No access to full-text legal databases**: Real legal AI (like Westlaw or LexisNexis) has access to millions of cases with full text. My dataset is limited to what's available on Kaggle.

2. **LLM can still reason incorrectly**: RAG grounds the output in real cases, but the LLM's reasoning might still be flawed. Legal decisions need human expert review.

3. **Chunking loses some context**: When we split cases into chunks, we might separate facts that are logically connected. Better chunking strategies (graph-based, hierarchical) would improve retrieval.

4. **No feedback loop**: The system doesn't learn from whether its predictions were actually correct. Adding an active learning loop would improve it over time.

---

## 💡 Numbers to Remember

- ECHR Dataset: **~11,000** cases
- Embedding dimension: **384** (all-MiniLM-L6-v2 model)
- Typical retrieval time: **< 100ms** (FAISS is very fast)
- Total prediction time (with LLM): **1-3 seconds**
- Context window used: Top **5-8** retrieved cases per query

---

## 🏆 Why This Project Stands Out

1. **Real-world domain**: Legal AI is a high-stakes, fast-growing field
2. **Production-grade**: Not a notebook — it's a REST API with Docker, CI/CD, and experiment tracking
3. **Multi-modal retrieval**: Combines semantic search (FAISS) with graph-based retrieval (NetworkX) — most projects use only one
4. **Explainable AI**: Every prediction comes with cited cases and reasoning, not just a label
5. **MLOps practices**: MLflow tracking, GitHub Actions CI, Docker deployment
