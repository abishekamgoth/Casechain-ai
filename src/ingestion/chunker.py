"""
src/ingestion/chunker.py
==========================
PURPOSE: Split large legal documents into smaller chunks for embedding.

WHY DO WE NEED CHUNKING?
  - An embedding model can only handle ~512 tokens at once (about 400 words)
  - A single legal case can be 10,000+ words
  - So we need to SPLIT each document into smaller pieces
  - But we can't split randomly! Legal documents have structure:
    → "Facts" section (what happened)
    → "Arguments" section (what each side argued)
    → "Judgment" section (what the court decided)
  - We do SECTION-AWARE splitting: we keep each section intact,
    then chunk within sections if they're still too long.

ANALOGY:
  Think of a legal case like a book with chapters.
  You wouldn't cut a sentence in half to fit a chapter.
  You'd split at chapter boundaries, then at paragraph boundaries.
  That's exactly what this chunker does.

OUTPUT:
  For each document, we produce multiple chunks like:
  [
    {"chunk_id": "ECHR-001-facts-0", "text": "The applicant was...", "section": "facts", ...},
    {"chunk_id": "ECHR-001-facts-1", "text": "In March 2019...", "section": "facts", ...},
    {"chunk_id": "ECHR-001-judgment-0", "text": "The Court finds...", "section": "judgment", ...},
  ]
"""

import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LegalDocumentChunker:
    """
    Section-aware document chunker for legal texts.
    
    Why "section-aware"?
    - Generic chunkers split every N characters (like cutting with scissors)
    - Ours first identifies the logical sections of a legal document
    - Then splits within sections, preserving context
    
    This is important because:
    - A user asking about FACTS should retrieve facts chunks
    - A user asking about OUTCOME should retrieve judgment chunks
    - Mixing them reduces retrieval quality
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Maximum number of words per chunk (not characters, words)
            chunk_overlap: How many words to repeat between adjacent chunks
                          (overlap = context continuity between chunks)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Regex patterns to detect section headers in legal text
        # Legal documents often have these sections
        self.section_patterns = {
            "facts": [
                r'\bFACTS?\b', r'\bBACKGROUND\b', r'\bCIRCUMSTANCES\b',
                r'\bTHE FACTS\b', r'\bSTATEMENT OF FACTS\b'
            ],
            "arguments": [
                r'\bARGUMENTS?\b', r'\bSUBMISSIONS?\b', r'\bALLEGATIONS?\b',
                r'\bTHE LAW\b', r'\bLEGAL ANALYSIS\b'
            ],
            "judgment": [
                r'\bJUDGMENT\b', r'\bDECISION\b', r'\bCONCLUSION\b',
                r'\bHELD\b', r'\bORDER\b', r'\bDISPOSITIF\b'
            ]
        }
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Main function: takes one legal document, returns a list of chunks.
        
        Process:
        1. Combine facts + judgment text
        2. Try to detect section boundaries
        3. Split each section into word-bounded chunks
        4. Add metadata (case_id, section type, chunk index) to each chunk
        
        Args:
            document: A unified document dict from DocumentLoader
            
        Returns:
            List of chunk dicts, each with text + metadata
        """
        chunks = []
        
        # Combine facts and judgment into one text for analysis
        full_text = f"{document.get('facts', '')} {document.get('judgment', '')}"
        
        # Try to detect and split by sections
        sections = self._detect_sections(full_text)
        
        if not sections:
            # If no clear sections detected, treat the whole text as "general"
            sections = {"general": full_text}
        
        # Now chunk each section separately
        for section_name, section_text in sections.items():
            if not section_text.strip():
                continue  # Skip empty sections
            
            section_chunks = self._split_into_chunks(section_text)
            
            for chunk_idx, chunk_text in enumerate(section_chunks):
                chunk = {
                    # Unique ID for this chunk
                    "chunk_id": f"{document['case_id']}-{section_name}-{chunk_idx}",
                    
                    # The actual text that will be embedded
                    "text": chunk_text,
                    
                    # Metadata stored alongside the embedding in FAISS
                    "case_id": document["case_id"],
                    "case_title": document.get("title", ""),
                    "section": section_name,       # "facts", "judgment", "general"
                    "outcome": document.get("outcome", "UNKNOWN"),
                    "articles": document.get("articles", []),
                    "citations": document.get("citations", []),
                    "source": document.get("source", ""),
                    "chunk_index": chunk_idx,
                }
                chunks.append(chunk)
        
        return chunks
    
    def chunk_all_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process ALL documents and return ALL chunks combined.
        This is what you call to prepare the full dataset for FAISS indexing.
        
        Args:
            documents: List of documents from DocumentLoader
            
        Returns:
            Flat list of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            try:
                doc_chunks = self.chunk_document(doc)
                all_chunks.extend(doc_chunks)
            except Exception as e:
                logger.error(f"Error chunking document {doc.get('case_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Chunking complete: {len(documents)} documents → {len(all_chunks)} chunks")
        return all_chunks
    
    def _detect_sections(self, text: str) -> Dict[str, str]:
        """
        Try to detect "Facts", "Arguments", and "Judgment" sections in text.
        
        Legal documents often look like:
        "THE FACTS
        The applicant was arrested...
        THE LAW
        The Court considers..."
        
        We find these headers and split the text at those points.
        
        Returns:
            Dict like {"facts": "...", "arguments": "...", "judgment": "..."}
            or empty dict if no clear sections found
        """
        sections = {}
        
        # Find positions of all section headers
        section_positions = []
        
        for section_name, patterns in self.section_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    section_positions.append({
                        "section": section_name,
                        "start": match.start(),
                        "end": match.end()
                    })
        
        if not section_positions:
            return {}  # No sections detected
        
        # Sort by position in the document
        section_positions.sort(key=lambda x: x["start"])
        
        # Extract text between section headers
        for i, section_info in enumerate(section_positions):
            start = section_info["end"]
            # End is either the start of the next section, or end of text
            end = section_positions[i + 1]["start"] if i + 1 < len(section_positions) else len(text)
            
            section_text = text[start:end].strip()
            section_name = section_info["section"]
            
            # If we've seen this section before, append to it
            if section_name in sections:
                sections[section_name] += " " + section_text
            else:
                sections[section_name] = section_text
        
        return sections
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into word-bounded chunks of self.chunk_size words,
        with self.chunk_overlap words of overlap between chunks.
        
        WHY OVERLAP?
        Imagine chunk 1 ends with "The court considered the..."
        and chunk 2 starts with "...detention to be unlawful."
        Without overlap, neither chunk makes full sense.
        With overlap, the boundary phrase appears in both chunks.
        
        Args:
            text: Raw text string
            
        Returns:
            List of text chunks (strings)
        """
        words = text.split()
        
        if len(words) <= self.chunk_size:
            # Text is short enough to be one chunk
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            
            # Move forward by (chunk_size - overlap) to create the overlap effect
            start += self.chunk_size - self.chunk_overlap
        
        return chunks


# ---- Test this chunker ----
if __name__ == "__main__":
    from src.ingestion.document_loader import DocumentLoader
    
    loader = DocumentLoader()
    documents = loader.load_all()
    
    chunker = LegalDocumentChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_all_documents(documents)
    
    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
    print(f"\n📄 Sample chunk:")
    import json
    print(json.dumps(chunks[0], indent=2))
