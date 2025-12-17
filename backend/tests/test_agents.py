"""
Unit tests for Agents
"""
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import SharedMemory
from agents.jd_agent import JDAgent

class TestAgents(unittest.TestCase):
    
    def setUp(self):
        self.shared_memory = SharedMemory()
        self.jd_agent = JDAgent(self.shared_memory)
    
    def test_shared_memory(self):
        self.shared_memory.set('test_key', 'test_value')
        value = self.shared_memory.get('test_key')
        self.assertEqual(value, 'test_value')
    
    def test_jd_agent_extract_skills(self):
        jd_text = "We are looking for a Python developer with experience in React and AWS."
        skills = self.jd_agent.extract_skills_from_jd(jd_text)
        self.assertIn('required_skills', skills)
        self.assertIn('optional_skills', skills)
    
    def test_jd_agent_process(self):
        result = self.jd_agent.process({
            'job_description': 'Python developer with 3+ years experience. React preferred.'
        })
        self.assertEqual(result['status'], 'success')
        self.assertIn('job_requirements', result)

if __name__ == '__main__':
    unittest.main()





