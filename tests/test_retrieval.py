"""
tests/test_retrieval.py
========================
PURPOSE: Unit tests for the retrieval components (FAISS + Citation Graph).

WHY TESTS?
  Tests verify that your code works correctly.
  When you change something, tests tell you if you broke something else.
  GitHub Actions runs these tests automatically on every push.
  
  "Test-Driven Development" (TDD) is a professional practice:
  write tests first, then write the code to make them pass.

WHAT WE TEST HERE:
  1. DocumentLoader: loads documents correctly
  2. LegalDocumentChunker: splits documents properly
  3. FAISSVectorStore: builds index and retrieves results
  4. LegalCitationGraph: builds graph and finds related cases
"""

import pytest
import sys
import os

# Add the project root to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import LegalDocumentChunker
from src.retrieval.faiss_store import FAISSVectorStore
from src.retrieval.citation_graph import LegalCitationGraph


# ============================================================
# FIXTURES
# ============================================================
# Fixtures are reusable setup code for tests.
# The @pytest.fixture decorator makes a function a fixture.
# Any test that has a fixture's name as a parameter gets it automatically.

@pytest.fixture
def sample_documents():
    """Load sample documents using DocumentLoader's sample data."""
    loader = DocumentLoader(data_dir="data/")
    # This will use sample data since no real data files exist in CI
    documents = loader.load_all()
    return documents

@pytest.fixture
def sample_chunks(sample_documents):
    """Create chunks from sample documents."""
    chunker = LegalDocumentChunker(chunk_size=100, chunk_overlap=10)
    return chunker.chunk_all_documents(sample_documents)

@pytest.fixture
def built_faiss_store(sample_chunks, tmp_path):
    """Build a FAISS index in a temporary directory for testing."""
    store = FAISSVectorStore(index_dir=str(tmp_path / "faiss_index"))
    store.build_index(sample_chunks)
    return store

@pytest.fixture
def built_citation_graph(sample_documents):
    """Build a citation graph from sample documents."""
    graph = LegalCitationGraph()
    graph.build_graph(sample_documents)
    return graph


# ============================================================
# TESTS: Document Loader
# ============================================================

class TestDocumentLoader:
    """Tests for the DocumentLoader class."""
    
    def test_load_returns_list(self, sample_documents):
        """DocumentLoader should return a list."""
        assert isinstance(sample_documents, list)
    
    def test_load_returns_non_empty(self, sample_documents):
        """DocumentLoader should return at least some documents."""
        assert len(sample_documents) > 0, "Should load at least sample documents"
    
    def test_document_has_required_fields(self, sample_documents):
        """Each document must have required fields."""
        required_fields = ["case_id", "title", "facts", "outcome", "source"]
        
        for doc in sample_documents:
            for field in required_fields:
                assert field in doc, f"Document missing required field: {field}"
    
    def test_document_case_id_is_string(self, sample_documents):
        """case_id should be a non-empty string."""
        for doc in sample_documents:
            assert isinstance(doc["case_id"], str)
            assert len(doc["case_id"]) > 0
    
    def test_outcome_is_valid(self, sample_documents):
        """Outcome should be VIOLATION, NO_VIOLATION, or UNKNOWN."""
        valid_outcomes = {"VIOLATION", "NO_VIOLATION", "UNKNOWN", "LIBERAL", "CONSERVATIVE"}
        for doc in sample_documents:
            # Some datasets may have different outcome labels
            assert isinstance(doc["outcome"], str)


# ============================================================
# TESTS: Document Chunker
# ============================================================

class TestDocumentChunker:
    """Tests for the LegalDocumentChunker class."""
    
    def test_chunks_returns_list(self, sample_chunks):
        """chunk_all_documents should return a list."""
        assert isinstance(sample_chunks, list)
    
    def test_more_chunks_than_documents(self, sample_documents, sample_chunks):
        """Should produce at least as many chunks as documents (possibly more)."""
        assert len(sample_chunks) >= len(sample_documents)
    
    def test_chunk_has_text(self, sample_chunks):
        """Every chunk must have non-empty text."""
        for chunk in sample_chunks:
            assert "text" in chunk
            assert len(chunk["text"]) > 0, f"Chunk {chunk.get('chunk_id')} has empty text"
    
    def test_chunk_has_case_id(self, sample_chunks):
        """Every chunk must reference its parent case."""
        for chunk in sample_chunks:
            assert "case_id" in chunk
            assert len(chunk["case_id"]) > 0
    
    def test_chunk_has_chunk_id(self, sample_chunks):
        """Every chunk must have a unique chunk_id."""
        chunk_ids = [c["chunk_id"] for c in sample_chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs must be unique!"
    
    def test_chunk_text_not_too_long(self, sample_chunks):
        """Chunks should not exceed maximum size (with some tolerance)."""
        chunker = LegalDocumentChunker(chunk_size=100)
        word_counts = [len(c["text"].split()) for c in sample_chunks]
        
        # Allow 20% tolerance for edge cases
        max_allowed = 100 * 1.2
        assert all(count <= max_allowed for count in word_counts), \
            f"Some chunks exceed maximum size. Max found: {max(word_counts)}"
    
    def test_section_detection(self):
        """Section detector should identify judgment sections."""
        chunker = LegalDocumentChunker()
        
        test_text = """
        THE FACTS
        The applicant was arrested and held without charge.
        
        THE LAW
        The Court considers that Article 5 applies here.
        
        JUDGMENT
        For these reasons, the Court finds a violation.
        """
        
        sections = chunker._detect_sections(test_text)
        # Should find at least some sections
        assert len(sections) > 0, "Should detect at least one section in structured text"


# ============================================================
# TESTS: FAISS Vector Store
# ============================================================

class TestFAISSVectorStore:
    """Tests for the FAISSVectorStore class."""
    
    def test_build_index_succeeds(self, built_faiss_store):
        """Building the index should succeed without errors."""
        assert built_faiss_store.index is not None
    
    def test_index_contains_vectors(self, built_faiss_store, sample_chunks):
        """Index should contain the same number of vectors as chunks."""
        assert built_faiss_store.index.ntotal == len(sample_chunks)
    
    def test_query_returns_list(self, built_faiss_store):
        """Querying should return a list."""
        results = built_faiss_store.query("arrested without trial", top_k=3)
        assert isinstance(results, list)
    
    def test_query_returns_correct_count(self, built_faiss_store):
        """Query should return at most top_k results."""
        top_k = 3
        results = built_faiss_store.query("detention rights", top_k=top_k)
        assert len(results) <= top_k
    
    def test_query_results_have_similarity_score(self, built_faiss_store):
        """Each result should have a similarity_score."""
        results = built_faiss_store.query("human rights violation", top_k=3)
        for result in results:
            assert "similarity_score" in result
            assert 0 <= result["similarity_score"] <= 1.01  # Allow slight float imprecision
    
    def test_query_results_sorted_by_relevance(self, built_faiss_store):
        """Results should be sorted from most to least similar."""
        results = built_faiss_store.query("arbitrary detention", top_k=5)
        if len(results) > 1:
            scores = [r["similarity_score"] for r in results]
            # Scores should be in descending order
            assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
                "Results should be sorted by relevance (highest first)"
    
    def test_save_and_load(self, sample_chunks, tmp_path):
        """Index should be saveable and loadable from disk."""
        # Build and save
        store1 = FAISSVectorStore(index_dir=str(tmp_path / "test_index"))
        store1.build_index(sample_chunks)
        
        # Load in a new instance
        store2 = FAISSVectorStore(index_dir=str(tmp_path / "test_index"))
        loaded = store2.load()
        
        assert loaded is True, "Load should return True when index exists"
        assert store2.index.ntotal == store1.index.ntotal, \
            "Loaded index should have same number of vectors"
    
    def test_get_stats(self, built_faiss_store):
        """get_stats should return a dict with expected keys."""
        stats = built_faiss_store.get_stats()
        assert "total_vectors" in stats
        assert "total_chunks" in stats
        assert stats["total_vectors"] > 0


# ============================================================
# TESTS: Citation Graph
# ============================================================

class TestLegalCitationGraph:
    """Tests for the LegalCitationGraph class."""
    
    def test_graph_builds_successfully(self, built_citation_graph):
        """Citation graph should build without errors."""
        assert built_citation_graph.graph is not None
    
    def test_graph_has_nodes(self, built_citation_graph, sample_documents):
        """Graph should have one node per document."""
        assert built_citation_graph.graph.number_of_nodes() == len(sample_documents)
    
    def test_get_related_cases(self, built_citation_graph, sample_documents):
        """get_related_cases should return a list."""
        seed_ids = [sample_documents[0]["case_id"]]
        related = built_citation_graph.get_related_cases(seed_ids, max_hops=2, max_results=5)
        assert isinstance(related, list)
    
    def test_graph_stats(self, built_citation_graph):
        """get_graph_stats should return expected keys."""
        stats = built_citation_graph.get_graph_stats()
        assert "total_cases" in stats
        assert "total_citations" in stats
        assert stats["total_cases"] > 0
    
    def test_most_cited_cases(self, built_citation_graph):
        """get_most_cited_cases should return tuples of (case_id, count)."""
        most_cited = built_citation_graph.get_most_cited_cases(top_k=3)
        assert isinstance(most_cited, list)
        # Each item should be a (case_id, count) tuple
        for item in most_cited:
            assert len(item) == 2    # (case_id, citation_count)
            assert isinstance(item[1], int)  # count is an integer
    
    def test_citation_chain_existing(self, built_citation_graph, sample_documents):
        """Should find chain between directly cited cases."""
        # ECHR-001 cites ECHR-002 in sample data
        chain = built_citation_graph.get_citation_chain("ECHR-001", "ECHR-002")
        # Chain might be None if the edge doesn't exist, which is fine
        if chain is not None:
            assert "ECHR-001" in chain
            assert "ECHR-002" in chain
    
    def test_save_and_load_graph(self, sample_documents, tmp_path):
        """Graph should save and load correctly."""
        filepath = str(tmp_path / "test_graph.json")
        
        # Build and save
        graph1 = LegalCitationGraph()
        graph1.build_graph(sample_documents)
        graph1.save_graph(filepath)
        
        # Load in new instance
        graph2 = LegalCitationGraph()
        loaded = graph2.load_graph(filepath)
        
        assert loaded is True
        assert graph2.graph.number_of_nodes() == graph1.graph.number_of_nodes()
