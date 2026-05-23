"""
src/retrieval/citation_graph.py
================================
PURPOSE: Build and query a citation graph using NetworkX to find
         cases that are related through citation chains.

WHY A CITATION GRAPH?
  FAISS finds cases that are SEMANTICALLY similar (similar meaning).
  But in law, there's another kind of similarity: CITATION RELATIONSHIPS.
  
  Example:
    - Case A directly mentions Case B as a precedent
    - Case B mentions Case C
    - So: A → B → C forms a "citation chain"
  
  Even if Case C has completely different words from our query,
  it might still be highly relevant because it's the original precedent
  that all related cases trace back to.
  
  FAISS can't find this relationship. NetworkX can.
  Together, they give us much richer retrieval.

HOW NETWORKX WORKS HERE:
  - Each legal case = a NODE in the graph
  - Each citation (Case A cites Case B) = a DIRECTED EDGE from A to B
  - We can then run graph algorithms:
    → BFS/DFS to find cases N hops away
    → PageRank to find the most "important" (most cited) cases
    → Shortest path between two cases

ANALOGY:
  Think of academic papers. When a paper cites other papers,
  it creates a citation network. Google Scholar uses this to
  show you related papers even when the topics differ.
  We're doing the same for legal cases.
"""

import networkx as nx
import json
import os
from typing import List, Dict, Set, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LegalCitationGraph:
    """
    Builds and queries a directed graph of legal case citations.
    
    Node = one legal case (identified by case_id)
    Edge = one case citing another case (directed: from newer to older)
    
    Node attributes stored:
    - title: Human-readable case name
    - outcome: VIOLATION / NO_VIOLATION
    - articles: Which laws were involved
    - source: Which dataset (echr, scotus, etc.)
    """
    
    def __init__(self):
        """
        DiGraph = Directed Graph
        Direction matters: Case A cites Case B means A → B (not B → A)
        """
        self.graph = nx.DiGraph()
    
    def build_graph(self, documents: List[Dict]) -> None:
        """
        Build the citation graph from all loaded documents.
        
        Two steps:
        1. Add all cases as nodes (with their metadata)
        2. Add all citation relationships as directed edges
        
        Args:
            documents: List of document dicts from DocumentLoader
        """
        logger.info("Building citation graph...")
        
        # --- Step 1: Add all cases as nodes ---
        for doc in documents:
            self.graph.add_node(
                doc["case_id"],
                title=doc.get("title", ""),
                outcome=doc.get("outcome", "UNKNOWN"),
                articles=doc.get("articles", []),
                source=doc.get("source", ""),
            )
        
        # --- Step 2: Add citation edges ---
        # If document A has ["ECHR-002", "ECHR-003"] in its citations field,
        # we add edges: A → ECHR-002 and A → ECHR-003
        edge_count = 0
        for doc in documents:
            citing_case = doc["case_id"]
            for cited_case_id in doc.get("citations", []):
                # Only add edge if the cited case exists in our dataset
                if self.graph.has_node(cited_case_id):
                    self.graph.add_edge(citing_case, cited_case_id)
                    edge_count += 1
        
        logger.info(
            f"Citation graph built: {self.graph.number_of_nodes()} cases, "
            f"{edge_count} citations"
        )
    
    def get_related_cases(
        self, 
        seed_case_ids: List[str], 
        max_hops: int = 2, 
        max_results: int = 10
    ) -> List[Dict]:
        """
        Given a list of seed cases, find cases related through citations.
        
        HOW IT WORKS:
        - Start at the seed cases
        - Follow citation edges up to max_hops steps
        - Collect all cases visited
        - Rank them by how close they are to the seeds
        
        Args:
            seed_case_ids: Cases found by FAISS (our starting points)
            max_hops: How many citation steps to follow (2 = friend of friend)
            max_results: Maximum cases to return
            
        Returns:
            List of related case metadata dicts, sorted by relevance
        """
        visited = set()
        results = []
        
        # BFS (Breadth-First Search) from each seed case
        for seed_id in seed_case_ids:
            if not self.graph.has_node(seed_id):
                continue
            
            # Find all cases within max_hops of this seed
            # nx.ego_graph returns the "neighborhood" of a node up to radius hops
            subgraph = nx.ego_graph(self.graph, seed_id, radius=max_hops)
            
            for node_id in subgraph.nodes():
                if node_id not in visited and node_id not in seed_case_ids:
                    visited.add(node_id)
                    
                    # Get hop distance from seed
                    try:
                        hops = nx.shortest_path_length(self.graph, seed_id, node_id)
                    except nx.NetworkXNoPath:
                        try:
                            hops = nx.shortest_path_length(self.graph, node_id, seed_id)
                        except nx.NetworkXNoPath:
                            hops = max_hops  # Default if no path found
                    
                    # Get node attributes from the graph
                    node_attrs = self.graph.nodes[node_id]
                    
                    results.append({
                        "case_id": node_id,
                        "title": node_attrs.get("title", ""),
                        "outcome": node_attrs.get("outcome", ""),
                        "articles": node_attrs.get("articles", []),
                        "source": node_attrs.get("source", ""),
                        "hop_distance": hops,
                        "relevance_type": "citation_graph",
                    })
        
        # Sort by hop distance (closer = more relevant)
        results.sort(key=lambda x: x["hop_distance"])
        
        return results[:max_results]
    
    def get_citation_chain(
        self, 
        from_case_id: str, 
        to_case_id: str
    ) -> Optional[List[str]]:
        """
        Find the citation path between two cases.
        
        Example: get_citation_chain("ECHR-001", "ECHR-005")
        Returns: ["ECHR-001", "ECHR-003", "ECHR-005"]
        This means: ECHR-001 cited ECHR-003, which cited ECHR-005
        
        Returns:
            List of case IDs forming the path, or None if no path exists
        """
        try:
            path = nx.shortest_path(self.graph, from_case_id, to_case_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_most_cited_cases(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """
        Find the most frequently cited cases in the dataset.
        This is like finding the most "landmark" or "precedent-setting" cases.
        
        We use in-degree (how many edges point INTO a node) as the citation count.
        A case with high in-degree is cited by many other cases = important precedent.
        
        Args:
            top_k: Number of top cases to return
            
        Returns:
            List of (case_id, citation_count) tuples, sorted by count (highest first)
        """
        # in_degree_centrality measures how many other nodes cite each node
        in_degree = dict(self.graph.in_degree())
        sorted_cases = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)
        return sorted_cases[:top_k]
    
    def run_pagerank(self) -> Dict[str, float]:
        """
        Run PageRank on the citation graph.
        
        PageRank (the algorithm behind Google Search) scores each node by:
        - How many cases cite it
        - How important those citing cases are (recursive)
        
        A case with high PageRank is both widely cited AND cited by important cases.
        This gives a better importance measure than just counting citations.
        
        Returns:
            Dict of {case_id: pagerank_score}
        """
        return nx.pagerank(self.graph, alpha=0.85)
    
    def get_graph_stats(self) -> Dict:
        """Return statistics about the citation graph."""
        return {
            "total_cases": self.graph.number_of_nodes(),
            "total_citations": self.graph.number_of_edges(),
            "avg_citations_per_case": (
                self.graph.number_of_edges() / max(self.graph.number_of_nodes(), 1)
            ),
            "is_connected": nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
    
    def save_graph(self, filepath: str = "data/citation_graph.json") -> None:
        """Save the graph to a JSON file using node-link format."""
        graph_data = nx.node_link_data(self.graph)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
        logger.info(f"Citation graph saved to {filepath}")
    
    def load_graph(self, filepath: str = "data/citation_graph.json") -> bool:
        """Load a previously saved graph from JSON."""
        if not os.path.exists(filepath):
            return False
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        self.graph = nx.node_link_graph(graph_data, directed=True)
        logger.info(f"Loaded citation graph: {self.get_graph_stats()}")
        return True


# ---- Test ----
if __name__ == "__main__":
    from src.ingestion.document_loader import DocumentLoader
    
    loader = DocumentLoader()
    documents = loader.load_all()
    
    cg = LegalCitationGraph()
    cg.build_graph(documents)
    
    print(f"Graph stats: {cg.get_graph_stats()}")
    print(f"Most cited cases: {cg.get_most_cited_cases(top_k=3)}")
    
    # Test citation chain
    chain = cg.get_citation_chain("ECHR-001", "ECHR-005")
    if chain:
        print(f"Citation chain from ECHR-001 to ECHR-005: {' → '.join(chain)}")
