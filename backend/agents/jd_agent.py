"""
Job Description (JD) Agent
Analyzes job descriptions and extracts requirements
"""
from .base_agent import BaseAgent, SharedMemory
from typing import Dict, List
import re

class JDAgent(BaseAgent):
    """Agent responsible for analyzing job descriptions"""
    
    def __init__(self, shared_memory: SharedMemory = None):
        super().__init__("jd_agent", shared_memory)
        self.skill_keywords = [
            'python', 'java', 'javascript', 'react', 'node', 'sql', 'mongodb',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'linux',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch',
            'scikit-learn', 'pandas', 'numpy', 'flask', 'django', 'fastapi'
        ]
    
    def extract_skills_from_jd(self, jd_text: str) -> Dict[str, List[str]]:
        """Extract required and optional skills from job description"""
        jd_lower = jd_text.lower()
        
        required_skills = []
        optional_skills = []
        
        # Look for required skills section
        required_patterns = [
            r'required[:\-]?\s*skills?[:\-]?\s*([^\.]+)',
            r'must have[:\-]?\s*([^\.]+)',
            r'essential[:\-]?\s*skills?[:\-]?\s*([^\.]+)'
        ]
        
        for pattern in required_patterns:
            matches = re.findall(pattern, jd_lower, re.IGNORECASE | re.DOTALL)
            for match in matches:
                skills = re.split(r'[,;]', match)
                for skill in skills:
                    skill = skill.strip()
                    if skill and len(skill) > 2:
                        required_skills.append(skill)
        
        # Look for optional/preferred skills
        optional_patterns = [
            r'preferred[:\-]?\s*skills?[:\-]?\s*([^\.]+)',
            r'nice to have[:\-]?\s*([^\.]+)',
            r'optional[:\-]?\s*skills?[:\-]?\s*([^\.]+)'
        ]
        
        for pattern in optional_patterns:
            matches = re.findall(pattern, jd_lower, re.IGNORECASE | re.DOTALL)
            for match in matches:
                skills = re.split(r'[,;]', match)
                for skill in skills:
                    skill = skill.strip()
                    if skill and len(skill) > 2:
                        optional_skills.append(skill)
        
        # Also check for common skill keywords in the text
        for skill in self.skill_keywords:
            if skill.lower() in jd_lower:
                if skill not in required_skills and skill not in optional_skills:
                    # If mentioned in requirements section, it's required
                    if any(keyword in jd_lower for keyword in ['required', 'must', 'essential']):
                        required_skills.append(skill)
                    else:
                        optional_skills.append(skill)
        
        return {
            'required_skills': list(set(required_skills)),
            'optional_skills': list(set(optional_skills))
        }
    
    def extract_experience_requirement(self, jd_text: str) -> int:
        """Extract required years of experience"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'experience[:\-]?\s*(\d+)\+?\s*years?',
            r'minimum[:\-]?\s*(\d+)\+?\s*years?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE)
            if matches:
                return max([int(m) for m in matches])
        
        return 0
    
    def extract_degree_requirement(self, jd_text: str) -> str:
        """Extract required degree"""
        degree_patterns = [
            r'(bachelor|master|phd|doctorate|b\.?s\.?|m\.?s\.?|ph\.?d\.?)',
            r'(degree|diploma)\s+(?:in|required)'
        ]
        
        jd_lower = jd_text.lower()
        for pattern in degree_patterns:
            matches = re.findall(pattern, jd_lower, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    return matches[0][0]
                return matches[0]
        
        return ""
    
    def extract_certifications(self, jd_text: str) -> Dict[str, List[str]]:
        """Extract required and optional certifications"""
        jd_lower = jd_text.lower()
        
        required_certs = []
        optional_certs = []
        
        cert_patterns = [
            r'certified\s+([a-zA-Z\s]+)',
            r'certification[:\-]?\s*([a-zA-Z\s]+)',
            r'([A-Z]{2,10})\s+certification'
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, jd_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    cert = match[0].strip()
                else:
                    cert = match.strip()
                
                if 'required' in jd_lower or 'must' in jd_lower:
                    required_certs.append(cert)
                else:
                    optional_certs.append(cert)
        
        return {
            'required_certifications': list(set(required_certs)),
            'optional_certifications': list(set(optional_certs))
        }
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process job description and extract requirements
        Input: {'job_description': str}
        Output: {'job_requirements': Dict}
        """
        self.set_status("processing")
        
        try:
            jd_text = input_data.get('job_description', '')
            
            # Extract all requirements
            skills = self.extract_skills_from_jd(jd_text)
            experience = self.extract_experience_requirement(jd_text)
            degree = self.extract_degree_requirement(jd_text)
            certifications = self.extract_certifications(jd_text)
            
            job_requirements = {
                'required_skills': skills['required_skills'],
                'optional_skills': skills['optional_skills'],
                'required_experience': experience,
                'required_degree': degree,
                'required_certifications': certifications['required_certifications'],
                'optional_certifications': certifications['optional_certifications'],
                'raw_text': jd_text
            }
            
            # Publish results
            self.publish("job_requirements", job_requirements)
            
            self.results = {
                'job_requirements': job_requirements,
                'status': 'success'
            }
            
            self.set_status("completed")
            return self.results
            
        except Exception as e:
            self.set_status("error")
            return {
                'status': 'error',
                'error': str(e)
            }





