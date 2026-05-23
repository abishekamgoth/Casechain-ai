"""
src/ingestion/document_loader.py
==================================
PURPOSE: Load legal case documents from different datasets (ECHR, SCOTUS, Indian Kanoon)
         and convert them into a unified format that the rest of our pipeline can use.

WHY THIS EXISTS:
  - Different datasets have different column names and structures
  - We need ONE consistent format for the rest of the pipeline
  - We also do basic cleaning here (remove null values, fix encoding issues)

WHAT IS A "DOCUMENT" HERE?
  A document = one court case, stored as a Python dict with these keys:
  {
    "case_id": "ECHR-2019-001",
    "title": "Smith v. United Kingdom",
    "facts": "The applicant was detained without...",   ← what happened
    "judgment": "The Court finds a violation of...",    ← what the court decided
    "outcome": "VIOLATION",                             ← our prediction target
    "articles": ["Article 5", "Article 6"],             ← which law was involved
    "citations": ["ECHR-2017-045", "ECHR-2015-022"],   ← cases this case referenced
    "source": "echr"                                    ← which dataset it came from
  }
"""

import json
import os
import pandas as pd
from typing import List, Dict, Optional
import logging

# Set up logging so we can see what's happening when we run this
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads legal case documents from multiple datasets.
    
    Think of this class like a librarian who knows how to read
    books in different languages (ECHR format, SCOTUS format, etc.)
    and translates them all into the same language (our unified format).
    """
    
    def __init__(self, data_dir: str = "data/"):
        """
        Args:
            data_dir: Path to the folder where your dataset files are stored
        """
        self.data_dir = data_dir
        self.documents = []  # Will hold all loaded documents
    
    def load_echr_dataset(self, filename: str = "echr_cases.json") -> List[Dict]:
        """
        Load the European Court of Human Rights dataset.
        
        ECHR cases have this structure:
        - Each case is about someone suing a European country
        - Outcomes are "VIOLATION" or "NO_VIOLATION"
        - Cases reference specific Articles (Article 5 = right to liberty, etc.)
        
        The ECHR dataset from Kaggle (STE dataset) has columns like:
        'itemid', 'docname', 'importance', 'conclusion', 'applicability', 'text'
        
        Returns:
            List of unified document dicts
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"ECHR file not found at {filepath}. Using sample data.")
            return self._get_echr_sample_data()
        
        logger.info(f"Loading ECHR dataset from {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        documents = []
        for idx, case in enumerate(raw_data):
            try:
                # Extract and normalize the outcome
                # ECHR uses "violation" or "no-violation" in the conclusion field
                conclusion = case.get('conclusion', '').lower()
                outcome = "VIOLATION" if "violation" in conclusion else "NO_VIOLATION"
                
                # Extract article references from the conclusion text
                articles = self._extract_articles(conclusion)
                
                doc = {
                    "case_id": f"ECHR-{case.get('itemid', idx)}",
                    "title": case.get('docname', f'Case {idx}'),
                    "facts": case.get('text', '')[:2000],      # First 2000 chars = facts section
                    "judgment": case.get('text', '')[2000:],   # Rest = judgment
                    "outcome": outcome,
                    "articles": articles,
                    "citations": case.get('citations', []),     # May or may not exist
                    "source": "echr",
                    "importance": case.get('importance', 3)     # 1=highest, 4=lowest priority
                }
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing ECHR case {idx}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} ECHR cases.")
        return documents
    
    def load_scotus_dataset(self, filename: str = "scotus_cases.json") -> List[Dict]:
        """
        Load the US Supreme Court (SCOTUS) dataset.
        
        SCOTUS cases are structured differently from ECHR:
        - 'petitioner' and 'respondent' (the two parties)
        - 'decision_direction': "liberal" or "conservative" 
        - We map this to our outcome format
        
        Returns:
            List of unified document dicts
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"SCOTUS file not found at {filepath}. Using sample data.")
            return self._get_scotus_sample_data()
        
        logger.info(f"Loading SCOTUS dataset from {filepath}...")
        df = pd.read_json(filepath)
        
        documents = []
        for idx, row in df.iterrows():
            try:
                doc = {
                    "case_id": f"SCOTUS-{row.get('docket_number', idx)}",
                    "title": f"{row.get('petitioner', 'Unknown')} v. {row.get('respondent', 'Unknown')}",
                    "facts": str(row.get('facts', '')),
                    "judgment": str(row.get('majority_opinion', '')),
                    "outcome": str(row.get('decision_direction', 'UNKNOWN')).upper(),
                    "articles": [str(row.get('issue_area', ''))],
                    "citations": [],   # SCOTUS dataset may not have structured citations
                    "source": "scotus"
                }
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing SCOTUS case {idx}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} SCOTUS cases.")
        return documents
    
    def load_all(self) -> List[Dict]:
        """
        Load ALL datasets and combine them into one big list.
        This is the main function you call to get all documents.
        
        Returns:
            Combined list of all documents from all datasets
        """
        logger.info("Starting to load all legal datasets...")
        
        all_docs = []
        all_docs.extend(self.load_echr_dataset())
        all_docs.extend(self.load_scotus_dataset())
        
        self.documents = all_docs
        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs
    
    def _extract_articles(self, text: str) -> List[str]:
        """
        Extract article references from legal text.
        Example: "violation of Article 5" → ["Article 5"]
        
        This is a simple keyword-based extractor.
        In production, you could use spaCy NER for better extraction.
        """
        import re
        # Find patterns like "Article 5", "Art. 6", "Article 3 § 2"
        pattern = r'[Aa]rt(?:icle)?\.?\s*(\d+)(?:\s*§\s*\d+)?'
        matches = re.findall(pattern, text)
        return [f"Article {m}" for m in set(matches)]
    
    def _get_echr_sample_data(self) -> List[Dict]:
        """
        Returns a small set of realistic sample ECHR cases.
        This is used when the real dataset isn't downloaded yet.
        Allows the code to run and be tested without the full dataset.
        """
        return [
            {
                "case_id": "ECHR-001",
                "title": "Aksoy v. Turkey",
                "facts": "The applicant was detained by police and subjected to ill-treatment. He was kept in custody for 14 days without being brought before a judge. During this period he had no access to a lawyer.",
                "judgment": "The Court finds a violation of Article 3 (prohibition of torture) and Article 5 (right to liberty). Turkey failed to protect the applicant from ill-treatment.",
                "outcome": "VIOLATION",
                "articles": ["Article 3", "Article 5"],
                "citations": ["ECHR-002", "ECHR-005"],
                "source": "echr",
                "importance": 1
            },
            {
                "case_id": "ECHR-002",
                "title": "Ireland v. United Kingdom",
                "facts": "The Republic of Ireland alleged that the United Kingdom had used interrogation techniques on detainees in Northern Ireland that amounted to torture and inhuman treatment.",
                "judgment": "The Court found a violation of Article 3. The techniques used constituted inhuman and degrading treatment, though not torture.",
                "outcome": "VIOLATION",
                "articles": ["Article 3"],
                "citations": [],
                "source": "echr",
                "importance": 1
            },
            {
                "case_id": "ECHR-003",
                "title": "Steel and Morris v. United Kingdom",
                "facts": "The applicants distributed leaflets critical of McDonald's. They were sued for defamation and were denied legal aid to defend themselves in the civil proceedings.",
                "judgment": "The Court finds violation of Article 6 (right to fair trial) and Article 10 (freedom of expression). Denial of legal aid in complex proceedings violated the Convention.",
                "outcome": "VIOLATION",
                "articles": ["Article 6", "Article 10"],
                "citations": ["ECHR-002"],
                "source": "echr",
                "importance": 2
            },
            {
                "case_id": "ECHR-004",
                "title": "Hatton v. United Kingdom",
                "facts": "The applicants living near Heathrow Airport complained that night flights caused sleep disturbance and violated their right to private life.",
                "judgment": "The Court found no violation of Article 8. The Government had struck a fair balance between the economic interests of the country and the applicants private life.",
                "outcome": "NO_VIOLATION",
                "articles": ["Article 8"],
                "citations": [],
                "source": "echr",
                "importance": 2
            },
            {
                "case_id": "ECHR-005",
                "title": "Fox, Campbell and Hartley v. United Kingdom",
                "facts": "The three applicants were arrested under the Prevention of Terrorism Act. They were not informed of the specific reasons for their arrest at the time of arrest.",
                "judgment": "The Court finds a violation of Article 5(2) which requires detainees to be informed promptly of the reasons for arrest in a language they understand.",
                "outcome": "VIOLATION",
                "articles": ["Article 5"],
                "citations": [],
                "source": "echr",
                "importance": 1
            }
        ]
    
    def _get_scotus_sample_data(self) -> List[Dict]:
        """Sample SCOTUS cases for when real data is unavailable."""
        return [
            {
                "case_id": "SCOTUS-001",
                "title": "Miranda v. Arizona",
                "facts": "Ernesto Miranda was arrested and interrogated by police without being informed of his right to have an attorney present and his right against self-incrimination.",
                "judgment": "The Court held that statements obtained during custodial interrogation without informing suspects of their constitutional rights are inadmissible.",
                "outcome": "VIOLATION",
                "articles": ["Fifth Amendment", "Sixth Amendment"],
                "citations": [],
                "source": "scotus"
            }
        ]


# ---- Script entry point ----
# Run this file directly to test the loader: python -m src.ingestion.document_loader

if __name__ == "__main__":
    loader = DocumentLoader(data_dir="data/")
    documents = loader.load_all()
    
    print(f"\n✅ Successfully loaded {len(documents)} legal cases")
    print(f"\n📄 Sample document:")
    print(json.dumps(documents[0], indent=2))
