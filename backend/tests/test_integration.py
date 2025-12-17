"""
Integration tests for API endpoints
"""
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from fastapi.testclient import TestClient
except ImportError:
    # Fallback if TestClient not available
    TestClient = None
from app.main import app

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        if TestClient is None:
            self.skipTest("TestClient not available")
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('message', response.json())
    
    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'healthy')
    
    def test_analyze_job_description(self):
        response = self.client.post("/api/analyze-job-description", json={
            "job_description": "Python developer with 3+ years experience"
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('job_requirements', response.json())

if __name__ == '__main__':
    unittest.main()

