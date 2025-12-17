"""
Ranking Agent
Runs Best-First Search to produce ranked final list
"""
from .base_agent import BaseAgent, SharedMemory
from typing import Dict, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.best_first import BestFirstSearch

class RankingAgent(BaseAgent):
    """Agent responsible for ranking candidates using Best-First Search"""
    
    def __init__(self, shared_memory: SharedMemory = None):
        super().__init__("ranking_agent", shared_memory)
        self.search_engine = BestFirstSearch()
    
    def process(self, input_data: Dict) -> Dict:
        """
        Rank candidates using Best-First Search
        Input: {
            'candidates': List[Dict],
            'ml_scores': List[float],
            'csp_scores': List[float],
            'job_requirements': Dict (optional)
        }
        Output: {'ranked_candidates': List[Dict]}
        """
        self.set_status("processing")
        
        try:
            candidates = input_data.get('candidates', [])
            ml_scores = input_data.get('ml_scores', [])
            csp_scores = input_data.get('csp_scores', [])
            job_requirements = input_data.get('job_requirements', {})
            
            # Ensure all lists have same length
            min_length = min(len(candidates), len(ml_scores), len(csp_scores))
            candidates = candidates[:min_length]
            ml_scores = ml_scores[:min_length]
            csp_scores = csp_scores[:min_length]
            
            # Perform ranking
            ranked_list = self.search_engine.rank_candidates(
                candidates=candidates,
                ml_scores=ml_scores,
                csp_scores=csp_scores,
                job_requirements=job_requirements
            )
            
            result = {
                'ranked_candidates': ranked_list,
                'total_candidates': len(ranked_list),
                'status': 'success'
            }
            
            # Publish results
            self.publish("ranked_list", ranked_list)
            
            self.results = result
            self.set_status("completed")
            
            return result
            
        except Exception as e:
            self.set_status("error")
            return {
                'status': 'error',
                'error': str(e)
            }





