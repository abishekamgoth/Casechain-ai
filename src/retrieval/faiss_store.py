"""
src/retrieval/faiss_store.py
==============================
PURPOSE: Build and query a FAISS vector index for semantic search over legal cases.

WHAT IS HAPPENING HERE? (Simple Explanation)
  
  Step 1 - BUILD the index:
    - Take all text chunks from our legal cases
    - Convert each chunk into a "vector" (a list of ~384 numbers)
    - These numbers represent the MEANING of the text
    - Store all these vectors in FAISS (a super-fast search database)
    - Save the index to disk so we don't re-build it every time
    
  Step 2 - QUERY the index:
    - User gives us case facts: "arrested without trial for 6 months"
    - Convert that to a vector too
    - Ask FAISS: "which stored vectors are closest to this query vector?"
    - FAISS returns the top-K most similar chunks
    - We look up those chunk IDs and return the original case documents

WHY FAISS?
  - Regular database: searches by exact keywords (like CTRL+F)
  - FAISS: searches by meaning (even if the words are different)
  - Example: "detained without charge" and "arrested without trial" have
    different words but very similar meanings → FAISS finds this similarity

ANALOGY:
  Think of each document as a point in space.
  Similar documents are close together.
  FAISS finds the nearest neighbors to your query point.
  It's like finding the closest city on a map, but in 384-dimensional space.
"""

import os
import json
import pickle
import numpy as np
import faiss
import logging
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Manages the FAISS vector index for legal case retrieval.
    
    This class handles:
    1. Building the index from chunks (run once)
    2. Saving/loading the index from disk (so we don't rebuild every time)
    3. Querying the index for similar cases
    """
    
    # We use a pre-trained model specifically designed for legal text
    # "all-MiniLM-L6-v2" is a small, fast model good for semantic search
    # It produces 384-dimensional vectors
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384  # Output size of the model above
    
    def __init__(self, index_dir: str = "data/faiss_index"):
        """
        Args:
            index_dir: Where to save/load the FAISS index files
        """
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        
        self.index_path = os.path.join(index_dir, "legal_cases.index")
        self.metadata_path = os.path.join(index_dir, "chunk_metadata.pkl")
        
        # Load the embedding model
        # This downloads the model first time, then uses the cached version
        logger.info(f"Loading embedding model: {self.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(self.EMBEDDING_MODEL)
        
        # These will be set when we build or load the index
        self.index = None           # The FAISS index object
        self.chunk_metadata = []    # List of metadata dicts (parallel to index vectors)
    
    def build_index(self, chunks: List[Dict]) -> None:
        """
        Build the FAISS index from scratch using all document chunks.
        
        This is the most computationally expensive step.
        For 10,000 chunks, this may take 5-10 minutes.
        After building, save to disk so we never need to rebuild.
        
        Args:
            chunks: List of chunk dicts from LegalDocumentChunker
        """
        logger.info(f"Building FAISS index from {len(chunks)} chunks...")
        
        # Step 1: Extract just the text from each chunk
        texts = [chunk["text"] for chunk in chunks]
        
        # Step 2: Convert all texts to vectors (embeddings)
        # batch_size=64 means we process 64 texts at a time (memory-efficient)
        logger.info("Generating embeddings (this may take a few minutes)...")
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,   # Shows a progress bar in terminal
            convert_to_numpy=True     # FAISS needs numpy arrays, not tensors
        )
        
        # Step 3: Normalize embeddings to unit length
        # This makes cosine similarity equivalent to dot product (faster)
        faiss.normalize_L2(embeddings)
        
        # Step 4: Create the FAISS index
        # IndexFlatIP = "Inner Product" index (best for normalized vectors)
        # It's exact (not approximate), good for our dataset size
        self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        
        # Step 5: Add all embeddings to the index
        self.index.add(embeddings.astype(np.float32))
        
        # Step 6: Store metadata alongside the index
        # FAISS only stores vectors (numbers), not the original text
        # So we keep a parallel list of metadata
        self.chunk_metadata = chunks
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
        
        # Save to disk
        self._save()
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Find the top-K most similar chunks to a query text.
        
        This is called every time a user submits case facts.
        It's fast: typically < 100ms even for large indices.
        
        Args:
            query_text: The user's case description
            top_k: How many similar cases to return
            
        Returns:
            List of chunk metadata dicts, sorted by relevance (most relevant first)
            Each dict also has a "similarity_score" field added
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load() first.")
        
        # Step 1: Convert query to a vector
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Step 2: Search the FAISS index
        # Returns: distances (similarity scores) and indices (positions in the index)
        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            top_k
        )
        
        # Step 3: Map indices back to original chunk metadata
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for "not enough results"
                continue
            
            chunk = self.chunk_metadata[idx].copy()
            chunk["similarity_score"] = float(dist)  # Add the relevance score
            results.append(chunk)
        
        return results
    
    def _save(self) -> None:
        """Save the FAISS index and metadata to disk."""
        logger.info(f"Saving FAISS index to {self.index_path}...")
        faiss.write_index(self.index, self.index_path)
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.chunk_metadata, f)
        
        logger.info("✅ FAISS index saved successfully.")
    
    def load(self) -> bool:
        """
        Load a previously built index from disk.
        
        Returns:
            True if loaded successfully, False if no saved index found
        """
        if not os.path.exists(self.index_path):
            logger.warning("No saved FAISS index found. Please build it first.")
            return False
        
        logger.info("Loading FAISS index from disk...")
        self.index = faiss.read_index(self.index_path)
        
        with open(self.metadata_path, 'rb') as f:
            self.chunk_metadata = pickle.load(f)
        
        logger.info(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
        return True
    
    def get_stats(self) -> Dict:
        """Return statistics about the current index."""
        if self.index is None:
            return {"status": "not_built"}
        
        return {
            "total_vectors": self.index.ntotal,
            "total_chunks": len(self.chunk_metadata),
            "embedding_dim": self.EMBEDDING_DIM,
            "index_size_mb": os.path.getsize(self.index_path) / (1024 * 1024)
            if os.path.exists(self.index_path) else 0
        }


# ---- Script entry point ----
if __name__ == "__main__":
    import argparse
    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.chunker import LegalDocumentChunker
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build the FAISS index")
    parser.add_argument("--query", type=str, help="Test query the index")
    args = parser.parse_args()
    
    store = FAISSVectorStore()
    
    if args.build:
        # Load documents and chunk them
        loader = DocumentLoader()
        documents = loader.load_all()
        
        chunker = LegalDocumentChunker()
        chunks = chunker.chunk_all_documents(documents)
        
        # Build and save the index
        store.build_index(chunks)
        print(f"\n✅ Index stats: {store.get_stats()}")
    
    elif args.query:
        store.load()
        results = store.query(args.query, top_k=3)
        
        print(f"\n🔍 Top {len(results)} results for: '{args.query}'\n")
        for i, result in enumerate(results, 1):
            print(f"  [{i}] Case: {result['case_title']}")
            print(f"      Score: {result['similarity_score']:.4f}")
            print(f"      Outcome: {result['outcome']}")
            print(f"      Section: {result['section']}")
            print(f"      Text: {result['text'][:200]}...\n")
    else:
        print("Use --build to build index, or --query 'your query' to test")
