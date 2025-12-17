"""
Unit tests for CSP Solver
"""
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csp.solver import CSPSolver

class TestCSPSolver(unittest.TestCase):
    
    def setUp(self):
        self.solver = CSPSolver()
    
    def test_check_mandatory_skills(self):
        candidate_skills = ['python', 'java', 'react']
        required_skills = ['python', 'java']
        constraint = self.solver.check_mandatory_skills(candidate_skills, required_skills)
        self.assertTrue(constraint.satisfied)
        
        candidate_skills = ['python']
        required_skills = ['python', 'java', 'react']
        constraint = self.solver.check_mandatory_skills(candidate_skills, required_skills)
        self.assertFalse(constraint.satisfied)
    
    def test_check_required_experience(self):
        constraint = self.solver.check_required_experience(5, 3)
        self.assertTrue(constraint.satisfied)
        
        constraint = self.solver.check_required_experience(2, 5)
        self.assertFalse(constraint.satisfied)
    
    def test_evaluate_candidate(self):
        candidate_data = {
            'skills': ['python', 'java', 'react'],
            'experience': {'years': 5},
            'education': [{'degree': 'bachelor'}],
            'certifications': []
        }
        
        job_requirements = {
            'required_skills': ['python', 'java'],
            'required_experience': 3,
            'required_degree': 'bachelor',
            'required_certifications': []
        }
        
        result = self.solver.evaluate_candidate(candidate_data, job_requirements)
        self.assertIn('eligibility_score', result)
        self.assertGreaterEqual(result['eligibility_score'], 0.0)
        self.assertLessEqual(result['eligibility_score'], 1.0)

if __name__ == '__main__':
    unittest.main()





