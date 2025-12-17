"""
Resume Parsing & Preprocessing Module
Extracts text from PDFs/TXT, cleans it, and extracts structured information
"""
import re
import PyPDF2
from typing import Dict, List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import ssl

# Handle SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data

# Download NLTK resources with fallback - do this at module import
def _download_nltk_resource(resource_name, fallback_name=None):
    """Download NLTK resource with fallback"""
    try:
        # Try to find the resource
        try:
            nltk.data.find(f'tokenizers/{resource_name}')
            return True
        except LookupError:
            pass
        
        # Try downloading the primary resource
        try:
            nltk.download(resource_name, quiet=True)
            return True
        except Exception as e:
            # If primary fails and we have a fallback, try it
            if fallback_name:
                try:
                    nltk.download(fallback_name, quiet=True)
                    return True
                except Exception:
                    pass
        return False
    except Exception:
        return False

# Try punkt_tab first (newer), fallback to punkt
_punkt_available = _download_nltk_resource('punkt_tab', 'punkt')
if not _punkt_available:
    # Last resort: try punkt without punkt_tab
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass  # Continue without stopwords - will use empty set

class ResumeParser:
    def __init__(self):
        # Handle stopwords gracefully
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # If stopwords not available, use empty set
            self.stop_words = set()
            print("Warning: NLTK stopwords not available, continuing without them")
        
        self.skill_keywords = self._load_skill_keywords()
        
        # Test tokenizer availability
        try:
            # Try to use tokenizer to verify it works
            test_text = "This is a test."
            word_tokenize(test_text)
            sent_tokenize(test_text)
        except LookupError as e:
            print(f"Warning: NLTK tokenizer issue: {e}. Some text processing may be limited.")
        except Exception as e:
            print(f"Warning: Tokenizer error: {e}. Continuing with basic text processing.")
        
    def _load_skill_keywords(self) -> List[str]:
        """Load common technical skills keywords"""
        return [
            'python', 'java', 'javascript', 'react', 'node', 'sql', 'mongodb',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'linux',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch',
            'scikit-learn', 'pandas', 'numpy', 'flask', 'django', 'fastapi',
            'html', 'css', 'typescript', 'angular', 'vue', 'express',
            'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka',
            'ci/cd', 'jenkins', 'terraform', 'ansible', 'agile', 'scrum'
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
        return text
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-]', '', text)
        # Convert to lowercase
        text = text.lower()
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text"""
        try:
            words = word_tokenize(text)
        except (LookupError, Exception):
            # Fallback: simple word splitting if NLTK tokenizer fails
            words = re.findall(r'\b\w+\b', text)
        
        filtered_words = [w for w in words if w.lower() not in self.stop_words and w.isalnum()]
        return ' '.join(filtered_words)
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        found_skills = []
        text_lower = text.lower()
        
        for skill in self.skill_keywords:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        # Also look for patterns like "skills:", "proficient in", etc.
        skill_patterns = [
            r'skills?[:\-]?\s*([^\.]+)',
            r'proficient in[:\-]?\s*([^\.]+)',
            r'technologies?[:\-]?\s*([^\.]+)',
            r'expertise in[:\-]?\s*([^\.]+)'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Extract individual skills from the match
                skills = re.split(r'[,;]', match)
                for skill in skills:
                    skill = skill.strip()
                    if skill and len(skill) > 2:
                        found_skills.append(skill)
        
        return list(set(found_skills))  # Remove duplicates
    
    def extract_experience(self, text: str) -> Dict:
        """Extract work experience information"""
        experience = {
            'years': 0,
            'companies': [],
            'positions': []
        }
        
        # Extract years of experience
        year_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'experience[:\-]?\s*(\d+)\+?\s*years?'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                years = [int(m) for m in matches]
                experience['years'] = max(years) if years else 0
                break
        
        # Extract company names (basic pattern)
        company_pattern = r'(?:at|in|with)\s+([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Ltd|Corp)?)'
        companies = re.findall(company_pattern, text)
        experience['companies'] = list(set(companies[:5]))  # Limit to 5
        
        # Extract job positions
        position_keywords = ['engineer', 'developer', 'manager', 'analyst', 'specialist', 
                           'architect', 'consultant', 'lead', 'senior', 'junior']
        try:
            sentences = sent_tokenize(text)
        except (LookupError, Exception):
            # Fallback: split on sentence endings if NLTK fails
            sentences = re.split(r'[.!?]+\s+', text)
        
        for sentence in sentences:
            for keyword in position_keywords:
                if keyword.lower() in sentence.lower():
                    # Extract the position title
                    match = re.search(r'([A-Z][a-zA-Z\s]+(?:' + keyword + '))', sentence, re.IGNORECASE)
                    if match:
                        experience['positions'].append(match.group(1))
        
        experience['positions'] = list(set(experience['positions'][:10]))
        return experience
    
    def extract_education(self, text: str) -> List[Dict]:
        """Extract education information"""
        education = []
        
        degree_patterns = [
            r'(bachelor|master|phd|doctorate|b\.?s\.?|m\.?s\.?|ph\.?d\.?)\s+(?:of|in)?\s*([a-zA-Z\s]+)',
            r'([a-zA-Z\s]+)\s+(?:degree|diploma)'
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    degree_type, field = match
                else:
                    degree_type = match
                    field = ""
                
                education.append({
                    'degree': degree_type.strip(),
                    'field': field.strip() if field else ""
                })
        
        # Extract university names
        university_keywords = ['university', 'college', 'institute', 'school']
        try:
            sentences = sent_tokenize(text)
        except (LookupError, Exception):
            # Fallback: split on sentence endings if NLTK fails
            sentences = re.split(r'[.!?]+\s+', text)
        
        for sentence in sentences:
            for keyword in university_keywords:
                if keyword.lower() in sentence.lower():
                    match = re.search(r'([A-Z][a-zA-Z\s&]+(?:' + keyword + '))', sentence, re.IGNORECASE)
                    if match:
                        education.append({'university': match.group(1)})
        
        return education[:5]  # Limit to 5 entries
    
    def extract_certifications(self, text: str) -> List[str]:
        """Extract certifications"""
        certifications = []
        
        cert_patterns = [
            r'certified\s+([a-zA-Z\s]+)',
            r'certification[:\-]?\s*([a-zA-Z\s]+)',
            r'([A-Z]{2,10})\s+certification',
            r'([A-Z]{2,10})\s+certified'
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend([m.strip() for m in matches])
        
        return list(set(certifications))
    
    def extract_projects(self, text: str) -> List[Dict]:
        """Extract project information"""
        projects = []
        
        # Look for project section
        project_section = re.search(r'projects?[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)', text, re.IGNORECASE | re.DOTALL)
        if project_section:
            project_text = project_section.group(1)
            # Split by common delimiters
            project_items = re.split(r'\n\s*[-•*]\s*|\n\d+\.\s*', project_text)
            
            for item in project_items[:10]:  # Limit to 10 projects
                if len(item.strip()) > 20:  # Only meaningful projects
                    # Try to extract project name
                    name_match = re.search(r'^([A-Z][^:]+)', item)
                    project_name = name_match.group(1).strip() if name_match else "Project"
                    
                    projects.append({
                        'name': project_name[:50],
                        'description': item.strip()[:500]
                    })
        
        return projects
    
    def extract_name(self, text: str) -> str:
        """Extract candidate name from resume text"""
        # Name is typically at the beginning of the resume
        # Look for patterns like:
        # - First line with proper case (2-4 words, capitalized)
        # - Name patterns: "John Doe", "John M. Doe", etc.
        
        lines = text.split('\n')
        
        # Check first few lines for name (more lines to check)
        for i, line in enumerate(lines[:15]):
            line = line.strip()
            if not line or len(line) < 3 or len(line) > 50:  # Name shouldn't be too long
                continue
            
            # More flexible pattern: 1-4 words, each starting with capital letter
            # Allow for middle initials, hyphens, etc.
            name_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?)?(?:\s+[A-Z][a-z]+){0,2})(?:\s|$|[:\-])'
            match = re.match(name_pattern, line)
            if match:
                name = match.group(1).strip()
                # Exclude common non-name patterns
                exclude_words = ['email', 'phone', 'address', 'resume', 'cv', 'objective', 'summary', 
                               'linkedin', 'github', 'portfolio', 'website', 'contact', 'mobile']
                if not any(word.lower() in exclude_words for word in name.split()):
                    # Check if it's not an email or phone
                    if '@' not in name and not re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', name):
                        # Check if it looks like a name (not all caps, not all lowercase)
                        if not name.isupper() and not name.islower():
                            return name
        
        # Fallback: Look for "Name:" pattern (case insensitive, more flexible)
        name_label_patterns = [
            r'(?:name|full\s+name|applicant)[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?)?(?:\s+[A-Z][a-z]+)?)',
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s|$)',  # Simple two-word name at start
        ]
        for pattern in name_label_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                if len(name) > 2 and len(name) < 50:
                    return name
        
        # Last resort: return first meaningful capitalized line (more flexible)
        for line in lines[:10]:
            line = line.strip()
            if len(line) < 3 or len(line) > 50:
                continue
            words = line.split()
            # Allow 1-4 words, at least first word capitalized
            if 1 <= len(words) <= 4 and words[0][0].isupper():
                # Check it's not an email or phone
                if '@' not in line and not re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', line):
                    return line
        
        return "Unknown"  # Default if name not found
    
    def parse_resume(self, file_path: str, file_type: str = 'pdf') -> Dict:
        """Main method to parse resume and extract all information"""
        # Extract text
        if file_type.lower() == 'pdf':
            raw_text = self.extract_text_from_pdf(file_path)
        else:
            raw_text = self.extract_text_from_txt(file_path)
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        text_without_stopwords = self.remove_stopwords(cleaned_text)
        
        # Extract structured information
        resume_data = {
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'name': self.extract_name(raw_text),  # Extract candidate name
            'skills': self.extract_skills(cleaned_text),
            'experience': self.extract_experience(cleaned_text),
            'education': self.extract_education(cleaned_text),
            'certifications': self.extract_certifications(cleaned_text),
            'projects': self.extract_projects(cleaned_text),
            'text_length': len(cleaned_text),
            'word_count': len(text_without_stopwords.split())
        }
        
        return resume_data





