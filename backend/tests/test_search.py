"""
Unit tests for Best-First Search
"""
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.best_first import BestFirstSearch

class TestBestFirstSearch(unittest.TestCase):
    
    def setUp(self):
        self.search = BestFirstSearch()
    
    def test_heuristic(self):
        from search.best_first import CandidateNode
        
        node = CandidateNode(
            candidate_id="test",
            candidate_data={'skills': ['python', 'java']},
            ml_score=0.8,
            csp_score=0.9
        )
        
        heuristic = self.search.heuristic(node)
        self.assertGreater(heuristic, 0)
    
    def test_rank_candidates(self):
        candidates = [
            {'candidate_id': '1', 'skills': ['python', 'java']},
            {'candidate_id': '2', 'skills': ['react', 'node']}
        ]
        ml_scores = [0.8, 0.6]
        csp_scores = [0.9, 0.7]
        
        ranked = self.search.rank_candidates(candidates, ml_scores, csp_scores)
        self.assertEqual(len(ranked), 2)
        self.assertIn('rank', ranked[0])

if __name__ == '__main__':
    unittest.main()





