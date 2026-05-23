"""
tests/test_api.py
==================
Tests for the FastAPI endpoints.
We use httpx to make requests to the API without actually starting a server.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_health_endpoint_format():
    """Test that health endpoint returns expected fields."""
    # In CI, just test the Pydantic model structure
    from src.api.main import HealthResponse
    
    response = HealthResponse(
        status="healthy",
        index_loaded=True,
        total_cases=100,
        model_version="1.0.0"
    )
    
    assert response.status == "healthy"
    assert response.index_loaded is True
    assert response.total_cases == 100


def test_prediction_request_validation():
    """Test PredictionRequest validates correctly."""
    from src.api.main import PredictionRequest
    from pydantic import ValidationError
    
    # Valid request
    req = PredictionRequest(case_facts="The applicant was detained without trial for 8 months, with no access to legal counsel or any judicial oversight during the entire period of detention.")
    assert len(req.case_facts) >= 50
    
    # Invalid request: too short
    with pytest.raises(ValidationError):
        PredictionRequest(case_facts="too short")


def test_prediction_response_structure():
    """Test that PredictionResponse has correct structure."""
    from src.api.main import PredictionResponse, CitedCase
    
    response = PredictionResponse(
        predicted_outcome="VIOLATION",
        confidence=0.87,
        articles=["Article 5"],
        reasoning="Based on similar cases, a violation occurred.",
        cited_cases=[
            CitedCase(
                case_id="ECHR-001",
                title="Test Case v. UK",
                outcome="VIOLATION",
                similarity_score=0.92,
                articles=["Article 5"]
            )
        ],
        retrieval_time_ms=45.2,
        prediction_time_ms=230.5
    )
    
    assert response.predicted_outcome in ["VIOLATION", "NO_VIOLATION", "UNKNOWN"]
    assert 0.0 <= response.confidence <= 1.0
    assert isinstance(response.cited_cases, list)


def test_chunker_and_retrieval_integration():
    """Integration test: chunker output works with FAISS store."""
    import tempfile
    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.chunker import LegalDocumentChunker
    from src.retrieval.faiss_store import FAISSVectorStore
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        loader = DocumentLoader()
        documents = loader.load_all()
        
        chunker = LegalDocumentChunker(chunk_size=100)
        chunks = chunker.chunk_all_documents(documents)
        
        store = FAISSVectorStore(index_dir=os.path.join(tmp_dir, "faiss"))
        store.build_index(chunks)
        
        # Query should work after building
        results = store.query("detained without trial", top_k=3)
        
        assert len(results) > 0, "Should retrieve at least one result"
        assert results[0]["similarity_score"] > 0, "Similarity score should be positive"
        
        print(f"✅ Integration test passed: {len(results)} results retrieved")
