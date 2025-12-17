"""
Best-First Search Algorithm Module
Implements graph-based search with priority queue for candidate ranking
"""
import heapq
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
import networkx as nx

@dataclass
class CandidateNode:
    """Represents a candidate node in the search graph"""
    candidate_id: str
    candidate_data: Dict
    ml_score: float = 0.0
    csp_score: float = 0.0
    heuristic_value: float = 0.0
    path_cost: float = 0.0
    
    def __lt__(self, other):
        """For priority queue ordering (higher heuristic = higher priority)"""
        return self.heuristic_value > other.heuristic_value

class BestFirstSearch:
    """Best-First Search implementation for candidate ranking"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.candidates = {}
        
    def build_graph(self, candidates: List[Dict], 
                   ml_scores: List[float],
                   csp_scores: List[float]):
        """Build graph of candidates with edges based on similarity"""
        self.graph.clear()
        self.candidates = {}
        
        # Add nodes
        for i, (candidate, ml_score, csp_score) in enumerate(zip(candidates, ml_scores, csp_scores)):
            # Use original candidate_id if available, otherwise generate one
            original_candidate_id = candidate.get('candidate_id', f"candidate_{i}")
            node_id = original_candidate_id  # Use original ID to preserve it
            
            # Preserve name and filename at top level of candidate_data for easy access
            candidate_with_metadata = candidate.copy()
            if 'name' in candidate:
                candidate_with_metadata['name'] = candidate['name']
            if 'filename' in candidate:
                candidate_with_metadata['filename'] = candidate['filename']
            
            self.candidates[node_id] = CandidateNode(
                candidate_id=original_candidate_id,  # Preserve original ID
                candidate_data=candidate_with_metadata,  # Include name and filename
                ml_score=ml_score,
                csp_score=csp_score
            )
            self.graph.add_node(node_id, 
                              ml_score=ml_score,
                              csp_score=csp_score,
                              candidate_data=candidate_with_metadata)
        
        # Add edges based on similarity (optional - for graph structure)
        # For ranking, we mainly use the heuristic function
        for i, node1_id in enumerate(self.candidates.keys()):
            for j, node2_id in enumerate(self.candidates.keys()):
                if i != j:
                    # Add edge with weight based on similarity
                    similarity = self._calculate_similarity(
                        self.candidates[node1_id].candidate_data,
                        self.candidates[node2_id].candidate_data
                    )
                    if similarity > 0.3:  # Threshold for edge creation
                        self.graph.add_edge(node1_id, node2_id, weight=similarity)
    
    def _calculate_similarity(self, candidate1: Dict, candidate2: Dict) -> float:
        """Calculate similarity between two candidates"""
        # Get skills from candidate_data if it's nested
        data1 = candidate1.get('candidate_data', candidate1) if 'candidate_data' in candidate1 else candidate1
        data2 = candidate2.get('candidate_data', candidate2) if 'candidate_data' in candidate2 else candidate2
        
        # Simple similarity based on skills overlap
        skills1 = set(s.lower() for s in data1.get('skills', []))
        skills2 = set(s.lower() for s in data2.get('skills', []))
        
        if not skills1 or not skills2:
            return 0.0
        
        intersection = skills1.intersection(skills2)
        union = skills1.union(skills2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def heuristic(self, node: CandidateNode, 
                 job_requirements: Optional[Dict] = None,
                 ml_weight: float = 0.6,
                 csp_weight: float = 0.4) -> float:
        """
        Calculate heuristic value for a candidate node
        Simplified: Use weighted combination of ML and CSP scores (same as Match-Agent)
        This ensures consistency with the Match-Agent's final_score calculation
        """
        # Use the same weighted combination as Match-Agent
        # final_score = (ML * 0.6) + (CSP * 0.4)
        heuristic_value = (node.ml_score * ml_weight) + (node.csp_score * csp_weight)
        
        return max(0.0, min(heuristic_value, 1.0))  # Clamp to [0, 1]
    
    def search(self, start_nodes: Optional[List[str]] = None,
              max_results: int = 100) -> List[Dict]:
        """
        Perform Best-First Search to rank candidates
        Returns ranked list of candidates
        """
        if not self.candidates:
            return []
        
        # Priority queue (min-heap, but we'll use negative values for max-heap behavior)
        priority_queue = []
        visited = set()
        
        # Initialize with all candidates or specified start nodes
        if start_nodes is None:
            start_nodes = list(self.candidates.keys())
        
        # Get job requirements from graph metadata if available
        job_requirements = getattr(self, 'job_requirements', None)
        
        # Calculate heuristic for all nodes and add to queue
        for node_id in start_nodes:
            if node_id in self.candidates:
                node = self.candidates[node_id]
                node.heuristic_value = self.heuristic(node, job_requirements)
                # Use negative for max-heap (Python's heapq is min-heap)
                heapq.heappush(priority_queue, (-node.heuristic_value, node_id, node))
        
        ranked_candidates = []
        
        # Best-First Search
        while priority_queue and len(ranked_candidates) < max_results:
            neg_heuristic, node_id, node = heapq.heappop(priority_queue)
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            
            # Extract candidate data (handle nested structure)
            candidate_data = node.candidate_data
            if isinstance(candidate_data, dict) and 'resume_data' in candidate_data:
                resume_data = candidate_data['resume_data']
            elif isinstance(candidate_data, dict) and 'candidate_data' in candidate_data:
                resume_data = candidate_data['candidate_data']
            else:
                resume_data = candidate_data
            
            # Add candidate to ranked list with all metadata
            candidate_info = {
                'candidate_id': node.candidate_id,  # Use original candidate_id
                'candidate_data': resume_data,  # Use extracted resume_data
                'ml_score': node.ml_score,
                'csp_score': node.csp_score,
                'heuristic_value': node.heuristic_value,
                'rank': len(ranked_candidates) + 1
            }
            
            # Include name and filename if available
            if 'name' in node.candidate_data:
                candidate_info['name'] = node.candidate_data['name']
            elif 'name' in resume_data:
                candidate_info['name'] = resume_data['name']
            
            if 'filename' in node.candidate_data:
                candidate_info['filename'] = node.candidate_data['filename']
            
            ranked_candidates.append(candidate_info)
            
            # Explore neighbors (if graph has edges)
            if node_id in self.graph:
                for neighbor_id in self.graph.neighbors(node_id):
                    if neighbor_id not in visited and neighbor_id in self.candidates:
                        neighbor_node = self.candidates[neighbor_id]
                        neighbor_node.heuristic_value = self.heuristic(neighbor_node, job_requirements)
                        heapq.heappush(priority_queue, 
                                     (-neighbor_node.heuristic_value, neighbor_id, neighbor_node))
        
        return ranked_candidates
    
    def rank_candidates(self, candidates: List[Dict],
                       ml_scores: List[float],
                       csp_scores: List[float],
                       job_requirements: Optional[Dict] = None) -> List[Dict]:
        """
        Main method to rank candidates - simplified to use final_score directly
        """
        # If candidates have final_score, use that for ranking (most accurate)
        if candidates and len(candidates) > 0 and 'final_score' in candidates[0]:
            # Create list with indices to preserve score mapping
            candidates_with_indices = [(idx, candidate) for idx, candidate in enumerate(candidates)]
            
            # Sort by final_score descending (highest first)
            candidates_with_indices.sort(key=lambda x: x[1].get('final_score', 0.0), reverse=True)
            
            # Build ranked list with all metadata
            ranked = []
            for rank_idx, (orig_idx, candidate) in enumerate(candidates_with_indices):
                ranked_candidate = candidate.copy()
                ranked_candidate['rank'] = rank_idx + 1
                ranked_candidate['ml_score'] = ml_scores[orig_idx] if orig_idx < len(ml_scores) else 0.0
                ranked_candidate['csp_score'] = csp_scores[orig_idx] if orig_idx < len(csp_scores) else 0.0
                ranked_candidate['heuristic_value'] = candidate.get('final_score', 0.0)  # Use final_score as heuristic
                
                # Ensure name and filename are preserved
                if 'name' not in ranked_candidate:
                    # Try to get from candidate_data or resume_data
                    candidate_data = ranked_candidate.get('candidate_data', {})
                    if isinstance(candidate_data, dict):
                        ranked_candidate['name'] = candidate_data.get('name', 'Unknown Candidate')
                
                if 'filename' not in ranked_candidate:
                    ranked_candidate['filename'] = 'Unknown File'
                
                ranked.append(ranked_candidate)
            
            return ranked
        
        # Fallback: Use Best-First Search with improved heuristic
        # Store job requirements for heuristic calculation
        self.job_requirements = job_requirements
        
        # Build graph
        self.build_graph(candidates, ml_scores, csp_scores)
        
        # Perform search
        ranked_list = self.search()
        
        return ranked_list





