"""
src/api/main.py
================
PURPOSE: The FastAPI REST API that exposes our RAG system as HTTP endpoints.

WHAT IS FASTAPI?
  FastAPI is a Python web framework (like Flask, but faster and modern).
  It lets us create API endpoints that other programs (or a frontend) can call.
  
  Example: When you call POST /predict with case facts,
  FastAPI receives the request, calls our RAG chain, and returns the prediction.

WHY AN API?
  - Makes our system usable from anywhere (web browser, mobile app, other software)
  - Standard interface: anyone who can make HTTP requests can use our system
  - Easy to deploy to cloud (Heroku, AWS, GCP, etc.)

ENDPOINTS IN THIS FILE:
  GET  /health               → Check if server is running
  POST /predict              → Main prediction (input: case facts, output: prediction)
  GET  /cases/search         → Search for similar cases
  GET  /citation-graph/{id}  → Get citation graph for a specific case
  GET  /stats                → System statistics (index size, total cases, etc.)

AUTOMATIC DOCUMENTATION:
  FastAPI auto-generates interactive API docs at:
  - http://localhost:8000/docs      (Swagger UI - you can test endpoints here!)
  - http://localhost:8000/redoc     (ReDoc - nicer looking docs)
"""

import os
import time
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow

# Our modules
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import LegalDocumentChunker
from src.retrieval.faiss_store import FAISSVectorStore
from src.retrieval.citation_graph import LegalCitationGraph
from src.prediction.rag_chain import LegalRAGChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# REQUEST / RESPONSE MODELS (Pydantic)
# ============================================================
# Pydantic models define the structure of API requests and responses.
# FastAPI uses these to:
# 1. Validate incoming data (wrong type = automatic 422 error)
# 2. Generate API documentation automatically
# 3. Serialize Python objects to JSON

class PredictionRequest(BaseModel):
    """What the user sends to /predict"""
    case_facts: str = Field(
        ...,                          # ... means "required"
        min_length=50,                # Must be at least 50 characters
        max_length=5000,
        description="Description of the legal case facts",
        example="The applicant was detained by police for 8 months without being brought before a judge. He had no access to a lawyer during this period."
    )
    top_k: int = Field(
        default=5,
        ge=1,                         # ge = "greater than or equal to"
        le=20,
        description="Number of similar cases to retrieve"
    )

class CitedCase(BaseModel):
    """One cited precedent case"""
    case_id: str
    title: str
    outcome: str
    similarity_score: float
    articles: List[str]

class PredictionResponse(BaseModel):
    """What we send back after /predict"""
    predicted_outcome: str           # "VIOLATION" or "NO_VIOLATION"
    confidence: float                # 0.0 to 1.0
    articles: List[str]              # Which laws are relevant
    reasoning: str                   # LLM's explanation
    cited_cases: List[CitedCase]     # Supporting precedents
    retrieval_time_ms: float         # How long retrieval took
    prediction_time_ms: float        # How long total prediction took

class SearchRequest(BaseModel):
    """Request for /cases/search"""
    query: str = Field(..., min_length=10, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

class HealthResponse(BaseModel):
    """Response from /health"""
    status: str
    index_loaded: bool
    total_cases: int
    model_version: str


# ============================================================
# APP INITIALIZATION
# ============================================================
# We use "lifespan" (new FastAPI way) to run setup code on startup.
# This loads our FAISS index and citation graph ONCE at startup,
# so they're ready for all subsequent requests.

# Global objects (loaded once, reused for all requests)
vector_store: Optional[FAISSVectorStore] = None
citation_graph: Optional[LegalCitationGraph] = None
rag_chain: Optional[LegalRAGChain] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    
    Code BEFORE yield = runs on startup
    Code AFTER yield = runs on shutdown
    """
    global vector_store, citation_graph, rag_chain
    
    logger.info("🚀 Starting Legal RAG API...")
    
    # Set up MLflow experiment tracking
    mlflow.set_experiment("legal-rag-predictions")
    
    # Load or build the vector store
    vector_store = FAISSVectorStore()
    index_loaded = vector_store.load()
    
    if not index_loaded:
        logger.info("No existing index found. Building from scratch...")
        loader = DocumentLoader()
        documents = loader.load_all()
        chunker = LegalDocumentChunker()
        chunks = chunker.chunk_all_documents(documents)
        vector_store.build_index(chunks)
    
    # Load the citation graph
    citation_graph = LegalCitationGraph()
    graph_loaded = citation_graph.load_graph()
    
    if not graph_loaded:
        logger.info("No existing citation graph found. Building...")
        loader = DocumentLoader()
        documents = loader.load_all()
        citation_graph.build_graph(documents)
        citation_graph.save_graph()
    
    # Initialize the RAG chain
    rag_chain = LegalRAGChain(
        vector_store=vector_store,
        citation_graph=citation_graph,
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        top_k_retrieval=5
    )
    
    logger.info("✅ Legal RAG API ready!")
    
    yield  # Application runs here
    
    # Shutdown: cleanup if needed
    logger.info("Shutting down Legal RAG API...")


# Create the FastAPI app
app = FastAPI(
    title="⚖️ Legal Case Prediction API",
    description="""
    RAG-based system for predicting legal case outcomes using past precedents.
    
    ## How it works
    1. Submit case facts via POST /predict
    2. System retrieves similar past cases from ECHR, SCOTUS, and Indian Kanoon datasets
    3. AI analyzes retrieved cases and predicts the outcome with citations
    
    Built with: LangChain, FAISS, NetworkX, FastAPI, MLflow
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware: allows the API to be called from web browsers
# (necessary for any frontend JavaScript to call your API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check if the API is running and ready.
    
    Use this to verify your deployment is working.
    Also tells you if the index is loaded and how many cases are available.
    """
    global vector_store
    
    if vector_store and vector_store.index:
        stats = vector_store.get_stats()
        return HealthResponse(
            status="healthy",
            index_loaded=True,
            total_cases=stats.get("total_chunks", 0),
            model_version="1.0.0"
        )
    else:
        return HealthResponse(
            status="degraded",
            index_loaded=False,
            total_cases=0,
            model_version="1.0.0"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_case_outcome(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    **Main endpoint**: Predict the outcome of a legal case.
    
    Given a description of case facts, this endpoint:
    1. Searches for similar past cases using semantic search (FAISS)
    2. Finds related cases through citation graph (NetworkX)
    3. Uses AI (Claude/GPT) to analyze retrieved cases and predict the outcome
    4. Returns prediction with confidence score and cited precedents
    
    **Input**: Case facts (50-5000 characters)  
    **Output**: VIOLATION or NO_VIOLATION with confidence, articles, and cited cases
    """
    global rag_chain
    
    if not rag_chain:
        raise HTTPException(
            status_code=503, 
            detail="RAG chain not initialized. Please wait for startup to complete."
        )
    
    # Time the overall prediction
    total_start = time.time()
    retrieval_start = time.time()
    
    try:
        result = rag_chain.predict(request.case_facts)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    retrieval_time = (time.time() - retrieval_start) * 1000
    total_time = (time.time() - total_start) * 1000
    
    # Log to MLflow in background (doesn't slow down the response)
    background_tasks.add_task(
        _log_to_mlflow,
        case_facts=request.case_facts,
        prediction=result,
        retrieval_time_ms=retrieval_time,
        total_time_ms=total_time
    )
    
    # Build and return the response
    cited_cases = [
        CitedCase(
            case_id=c.get("case_id", ""),
            title=c.get("title", "Unknown"),
            outcome=c.get("outcome", "UNKNOWN"),
            similarity_score=c.get("similarity_score", 0.0),
            articles=c.get("articles", [])
        )
        for c in result.get("cited_cases", [])
    ]
    
    return PredictionResponse(
        predicted_outcome=result["predicted_outcome"],
        confidence=result["confidence"],
        articles=result.get("articles", []),
        reasoning=result.get("reasoning", ""),
        cited_cases=cited_cases,
        retrieval_time_ms=retrieval_time,
        prediction_time_ms=total_time
    )


@app.get("/cases/search", tags=["Cases"])
async def search_similar_cases(query: str, top_k: int = 5):
    """
    Search for cases similar to a query string.
    
    This is a pure semantic search — no prediction, just retrieval.
    Useful for exploring the database or finding related cases.
    
    Example: GET /cases/search?query=detention without trial&top_k=5
    """
    global vector_store
    
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not ready")
    
    if len(query) < 5:
        raise HTTPException(status_code=400, detail="Query must be at least 5 characters")
    
    results = vector_store.query(query, top_k=min(top_k, 20))
    
    return {
        "query": query,
        "total_results": len(results),
        "cases": [
            {
                "case_id": r.get("case_id"),
                "title": r.get("case_title", ""),
                "outcome": r.get("outcome"),
                "articles": r.get("articles", []),
                "similarity_score": r.get("similarity_score"),
                "relevant_text": r.get("text", "")[:300] + "..."
            }
            for r in results
        ]
    }


@app.get("/citation-graph/{case_id}", tags=["Cases"])
async def get_citation_graph(case_id: str, max_hops: int = 2):
    """
    Get the citation graph for a specific case.
    
    Returns cases that cite, or are cited by, the given case
    up to max_hops levels deep.
    
    Example: GET /citation-graph/ECHR-001?max_hops=2
    """
    global citation_graph
    
    if not citation_graph:
        raise HTTPException(status_code=503, detail="Citation graph not ready")
    
    related = citation_graph.get_related_cases([case_id], max_hops=max_hops, max_results=20)
    
    return {
        "case_id": case_id,
        "max_hops": max_hops,
        "related_cases": related,
        "total_related": len(related)
    }


@app.get("/stats", tags=["System"])
async def get_system_stats():
    """
    Get statistics about the system.
    
    Returns:
    - Number of indexed cases/chunks
    - Citation graph statistics
    - Index file size
    """
    global vector_store, citation_graph
    
    stats = {}
    
    if vector_store:
        stats["vector_index"] = vector_store.get_stats()
    
    if citation_graph:
        stats["citation_graph"] = citation_graph.get_graph_stats()
    
    return stats


# ============================================================
# BACKGROUND TASKS
# ============================================================

async def _log_to_mlflow(
    case_facts: str,
    prediction: dict,
    retrieval_time_ms: float,
    total_time_ms: float
):
    """
    Log prediction metrics to MLflow in the background.
    
    WHY MLFLOW?
    - Tracks every prediction we make
    - We can analyze: which cases trigger violations? What's the average confidence?
    - Helps us improve the system over time
    - Required for the MLOps component of this project
    """
    try:
        with mlflow.start_run(run_name="prediction"):
            # Log inputs
            mlflow.log_param("case_facts_length", len(case_facts))
            mlflow.log_param("top_k", 5)
            
            # Log prediction outputs
            mlflow.log_metric("confidence", prediction.get("confidence", 0.0))
            mlflow.log_metric("retrieval_time_ms", retrieval_time_ms)
            mlflow.log_metric("total_time_ms", total_time_ms)
            mlflow.log_metric("num_cited_cases", len(prediction.get("cited_cases", [])))
            mlflow.log_metric("is_violation", 1 if prediction.get("predicted_outcome") == "VIOLATION" else 0)
            
            # Log the raw outcome as a tag
            mlflow.set_tag("predicted_outcome", prediction.get("predicted_outcome", "UNKNOWN"))
    
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-critical): {e}")


# ============================================================
# RUN THE APP
# ============================================================
# This block only runs when you execute: python src/api/main.py directly
# When using uvicorn (production), this doesn't execute.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True   # Auto-reload when you change the code (development mode)
    )
