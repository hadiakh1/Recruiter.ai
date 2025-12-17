"""
Unit tests for Resume Parser
"""
import unittest
import os
import tempfile
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.resume_parser import ResumeParser

class TestResumeParser(unittest.TestCase):
    
    def setUp(self):
        self.parser = ResumeParser()
    
    def test_clean_text(self):
        text = "  Hello   World  !!!  "
        cleaned = self.parser.clean_text(text)
        self.assertEqual(cleaned, "hello world")
    
    def test_extract_skills(self):
        text = "I have experience with Python, Java, and React. Also know Docker and AWS."
        skills = self.parser.extract_skills(text)
        self.assertGreater(len(skills), 0)
        self.assertIn('python', [s.lower() for s in skills])
    
    def test_extract_experience(self):
        text = "I have 5 years of experience in software development."
        experience = self.parser.extract_experience(text)
        self.assertGreaterEqual(experience['years'], 0)
    
    def test_extract_education(self):
        text = "I have a Bachelor of Science in Computer Science from MIT."
        education = self.parser.extract_education(text)
        self.assertGreater(len(education), 0)

if __name__ == '__main__':
    unittest.main()





