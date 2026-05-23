"""
src/prediction/rag_chain.py
=============================
PURPOSE: The core RAG (Retrieval-Augmented Generation) chain using LangChain.

THIS IS THE HEART OF THE PROJECT.

What happens when a user submits case facts:

  1. User input: "The applicant was arrested and held for 8 months without trial"
  
  2. RETRIEVAL:
     - FAISS finds the top-5 most semantically similar case chunks
     - Citation graph finds 3 more cases related through citations
     - We combine these 8 cases into a "context"
  
  3. GENERATION (the "G" in RAG):
     - We format a prompt: "Here are past cases: [context]. 
       Given these, predict the outcome of: [user input]"
     - We send this to Claude/GPT
     - The LLM reads the retrieved cases and generates a structured answer
     - The answer includes: predicted outcome + which cases support it + reasoning
  
  4. Why is this better than just asking an LLM directly?
     - Without RAG: LLM might hallucinate fake cases
     - With RAG: LLM can ONLY cite cases we retrieved (real, existing cases)
     - The answer is grounded in actual legal precedents

LANGCHAIN COMPONENTS USED:
  - PromptTemplate: Formats our custom legal prompt
  - RetrievalQA: Chains together retrieval + LLM call
  - Anthropic (Claude): The LLM that reads context and generates answers
"""

import os
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

load_dotenv()  # Load API keys from .env file
logger = logging.getLogger(__name__)

# LangChain imports
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever

# Our custom modules
from src.retrieval.faiss_store import FAISSVectorStore
from src.retrieval.citation_graph import LegalCitationGraph


# ============================================================
# LEGAL PROMPT TEMPLATE
# ============================================================
# This is carefully designed for legal reasoning.
# It tells the LLM:
# 1. What role to play (legal AI assistant)
# 2. What context to use (retrieved cases)
# 3. What format to respond in (structured JSON-like)

LEGAL_PROMPT_TEMPLATE = """
You are an expert legal AI assistant trained to analyze court cases and predict outcomes based on legal precedents.

RETRIEVED LEGAL PRECEDENTS:
{context}

NEW CASE FACTS:
{question}

Based ONLY on the retrieved precedents above, provide:
1. PREDICTED OUTCOME: (VIOLATION or NO_VIOLATION)
2. CONFIDENCE: (0.0 to 1.0, where 1.0 = very certain)
3. APPLICABLE ARTICLES: (Which laws or articles are relevant)
4. KEY PRECEDENTS: (List the 2-3 most relevant retrieved cases and why they apply)
5. REASONING: (2-3 sentences explaining why these precedents support your prediction)

If the retrieved cases do not contain enough information to make a confident prediction, 
say so explicitly and set confidence below 0.5.

Format your response as:
OUTCOME: [VIOLATION/NO_VIOLATION]
CONFIDENCE: [0.0-1.0]
ARTICLES: [comma-separated list]
PRECEDENTS: [case titles with one-line explanation each]
REASONING: [2-3 sentences]
"""


class LegalRAGChain:
    """
    The main RAG pipeline for legal case prediction.
    
    This class:
    1. Takes user's case facts as input
    2. Uses FAISS to retrieve similar cases
    3. Uses the citation graph to add related cases
    4. Formats everything into a legal prompt
    5. Sends to LLM (Claude/OpenAI)
    6. Parses and returns structured prediction
    """
    
    def __init__(
        self, 
        vector_store: FAISSVectorStore,
        citation_graph: LegalCitationGraph,
        llm_provider: str = "anthropic",  # "anthropic" or "openai"
        top_k_retrieval: int = 5,
    ):
        """
        Args:
            vector_store: Pre-built FAISS index
            citation_graph: Pre-built citation graph
            llm_provider: Which LLM to use for generation
            top_k_retrieval: How many cases to retrieve from FAISS
        """
        self.vector_store = vector_store
        self.citation_graph = citation_graph
        self.top_k = top_k_retrieval
        
        # Initialize the LLM
        self.llm = self._initialize_llm(llm_provider)
        
        # Create the prompt template
        self.prompt = PromptTemplate(
            template=LEGAL_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        logger.info(f"LegalRAGChain initialized with {llm_provider} LLM")
    
    def predict(self, case_facts: str) -> Dict:
        """
        Main prediction function. Given case facts, return a structured prediction.
        
        This is what the FastAPI endpoint calls.
        
        Args:
            case_facts: Description of the new legal case
            
        Returns:
            Dict with: outcome, confidence, articles, cited_cases, reasoning
        """
        logger.info(f"Processing case: {case_facts[:100]}...")
        
        # ---- Step 1: RETRIEVE from FAISS ----
        faiss_results = self.vector_store.query(case_facts, top_k=self.top_k)
        
        # ---- Step 2: ENRICH with Citation Graph ----
        seed_case_ids = list(set([r["case_id"] for r in faiss_results]))
        graph_results = self.citation_graph.get_related_cases(
            seed_case_ids, max_hops=2, max_results=3
        )
        
        # ---- Step 3: COMBINE and deduplicate results ----
        all_results = faiss_results + graph_results
        seen_cases = set()
        unique_results = []
        for r in all_results:
            if r["case_id"] not in seen_cases:
                seen_cases.add(r["case_id"])
                unique_results.append(r)
        
        # ---- Step 4: FORMAT context for the LLM ----
        context = self._format_context(unique_results)
        
        # ---- Step 5: CALL the LLM ----
        try:
            llm_response = self._call_llm(context, case_facts)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            llm_response = self._fallback_prediction(faiss_results)
        
        # ---- Step 6: PARSE LLM response into structured format ----
        parsed = self._parse_llm_response(llm_response, unique_results)
        
        return parsed
    
    def _initialize_llm(self, provider: str):
        """
        Initialize the LLM based on provider choice.
        
        Why support both Anthropic and OpenAI?
        - Flexibility: some users have one key but not the other
        - Cost: different pricing models
        - Capability: different strengths for different tasks
        """
        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not found. Falling back to OpenAI.")
                return self._initialize_llm("openai")
            
            return ChatAnthropic(
                model="claude-3-haiku-20240307",  # Fast and cheap, good for this task
                anthropic_api_key=api_key,
                max_tokens=1024,
                temperature=0.1,  # Low temperature = more deterministic, less creative
            )
        
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
            
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=api_key,
                temperature=0.1,
            )
        
        else:
            raise ValueError(f"Unknown LLM provider: {provider}. Use 'anthropic' or 'openai'")
    
    def _format_context(self, retrieved_cases: List[Dict]) -> str:
        """
        Format the retrieved cases into a readable context string for the LLM.
        
        The LLM reads this as its "legal library" to answer from.
        Better formatting = better LLM responses.
        """
        if not retrieved_cases:
            return "No relevant precedents found in database."
        
        context_parts = []
        
        for i, case in enumerate(retrieved_cases[:8], 1):  # Limit to 8 cases max
            # Format each case as a clear, structured block
            case_text = f"""
PRECEDENT {i}: {case.get('case_title', case.get('title', 'Unknown Case'))}
  - Case ID: {case.get('case_id', 'N/A')}
  - Outcome: {case.get('outcome', 'UNKNOWN')}
  - Articles Involved: {', '.join(case.get('articles', [])) or 'Not specified'}
  - Relevance Type: {case.get('relevance_type', 'semantic_similarity')}
  - Relevant Facts: {case.get('text', '')[:400]}...
"""
            context_parts.append(case_text)
        
        return "\n".join(context_parts)
    
    def _call_llm(self, context: str, question: str) -> str:
        """
        Send the formatted prompt to the LLM and get a response.
        """
        # Format the prompt with our context and question
        formatted_prompt = self.prompt.format(
            context=context,
            question=question
        )
        
        # Call the LLM
        response = self.llm.invoke(formatted_prompt)
        
        # Extract text from the response object
        return response.content if hasattr(response, 'content') else str(response)
    
    def _parse_llm_response(self, llm_text: str, retrieved_cases: List[Dict]) -> Dict:
        """
        Parse the LLM's text response into a structured Python dict.
        
        The LLM responds in our template format:
        OUTCOME: VIOLATION
        CONFIDENCE: 0.87
        ARTICLES: Article 5, Article 6
        PRECEDENTS: Smith v. UK - detention without charge...
        REASONING: Based on 3 similar cases...
        
        We extract each field using simple string parsing.
        """
        lines = llm_text.strip().split('\n')
        parsed = {
            "predicted_outcome": "UNKNOWN",
            "confidence": 0.5,
            "articles": [],
            "reasoning": "",
            "raw_llm_response": llm_text,
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith("OUTCOME:"):
                outcome = line.replace("OUTCOME:", "").strip()
                parsed["predicted_outcome"] = "VIOLATION" if "VIOLATION" in outcome.upper() else "NO_VIOLATION"
            
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.replace("CONFIDENCE:", "").strip()
                    parsed["confidence"] = float(conf_str)
                except ValueError:
                    parsed["confidence"] = 0.5
            
            elif line.startswith("ARTICLES:"):
                articles_str = line.replace("ARTICLES:", "").strip()
                parsed["articles"] = [a.strip() for a in articles_str.split(",") if a.strip()]
            
            elif line.startswith("REASONING:"):
                parsed["reasoning"] = line.replace("REASONING:", "").strip()
        
        # Add the top cited cases from retrieval
        parsed["cited_cases"] = [
            {
                "case_id": case.get("case_id"),
                "title": case.get("case_title", case.get("title", "")),
                "outcome": case.get("outcome"),
                "similarity_score": case.get("similarity_score", 0.0),
                "articles": case.get("articles", []),
            }
            for case in retrieved_cases[:5]  # Top 5 cited cases
        ]
        
        return parsed
    
    def _fallback_prediction(self, faiss_results: List[Dict]) -> str:
        """
        If LLM call fails, fall back to a simple majority-vote prediction.
        
        Look at retrieved cases, count VIOLATION vs NO_VIOLATION,
        and predict whichever is more common.
        This is a basic baseline that doesn't need an LLM.
        """
        if not faiss_results:
            return "OUTCOME: UNKNOWN\nCONFIDENCE: 0.0\nARTICLES: \nPRECEDENTS: None\nREASONING: Could not retrieve relevant cases."
        
        outcomes = [r.get("outcome", "") for r in faiss_results]
        violation_count = sum(1 for o in outcomes if o == "VIOLATION")
        no_violation_count = len(outcomes) - violation_count
        
        predicted = "VIOLATION" if violation_count > no_violation_count else "NO_VIOLATION"
        confidence = max(violation_count, no_violation_count) / len(outcomes)
        
        return (
            f"OUTCOME: {predicted}\n"
            f"CONFIDENCE: {confidence:.2f}\n"
            f"ARTICLES: Unknown\n"
            f"PRECEDENTS: Based on {len(faiss_results)} retrieved cases\n"
            f"REASONING: Majority vote from retrieved similar cases (LLM unavailable)."
        )
