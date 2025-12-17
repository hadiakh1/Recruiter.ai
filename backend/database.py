"""
Database module for SQLite storage
Handles users, resumes, and job descriptions
"""
import sqlite3
import os
from typing import Optional, Dict, List
from datetime import datetime
import json

DB_PATH = os.path.join(os.path.dirname(__file__), 'recruitment.db')

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Resumes table (linked to users)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            candidate_id TEXT UNIQUE NOT NULL,
            resume_data TEXT NOT NULL,
            features TEXT,
            filename TEXT,
            status TEXT DEFAULT 'processed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    # Add filename column if it doesn't exist (for existing databases)
    try:
        cursor.execute('ALTER TABLE resumes ADD COLUMN filename TEXT')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Job descriptions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            job_id TEXT UNIQUE NOT NULL,
            job_description TEXT NOT NULL,
            job_requirements TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    # Rankings table (store ranking results)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            job_id TEXT NOT NULL,
            ranking_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

def create_user(email: str, name: str, hashed_password: str) -> Optional[int]:
    """Create a new user"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (email, name, hashed_password) VALUES (?, ?, ?)',
            (email.lower(), name, hashed_password)
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None  # Email already exists
    except Exception as e:
        print(f"Error creating user: {e}")
        return None

def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email.lower(),))
        row = cursor.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None

def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Get user by ID"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None

def save_resume(user_id: int, candidate_id: str, resume_data: Dict, features: Optional[Dict] = None, filename: Optional[str] = None) -> bool:
    """Save resume data"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO resumes (user_id, candidate_id, resume_data, features, filename) VALUES (?, ?, ?, ?, ?)',
            (user_id, candidate_id, json.dumps(resume_data), json.dumps(features) if features else None, filename)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving resume: {e}")
        return False

def get_resume(candidate_id: str) -> Optional[Dict]:
    """Get resume by candidate_id"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM resumes WHERE candidate_id = ?', (candidate_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            data = dict(row)
            data['resume_data'] = json.loads(data['resume_data'])
            if data['features']:
                data['features'] = json.loads(data['features'])
            return data
        return None
    except Exception as e:
        print(f"Error getting resume: {e}")
        return None

def get_user_resumes(user_id: int) -> List[Dict]:
    """Get all resumes for a user"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM resumes WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
        rows = cursor.fetchall()
        conn.close()
        resumes = []
        for row in rows:
            data = dict(row)
            data['resume_data'] = json.loads(data['resume_data'])
            if data['features']:
                data['features'] = json.loads(data['features'])
            resumes.append(data)
        return resumes
    except Exception as e:
        print(f"Error getting user resumes: {e}")
        return []

def save_job_description(user_id: int, job_id: str, job_description: str, job_requirements: Dict) -> bool:
    """Save job description"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO job_descriptions (user_id, job_id, job_description, job_requirements) VALUES (?, ?, ?, ?)',
            (user_id, job_id, job_description, json.dumps(job_requirements))
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving job description: {e}")
        return False

def get_job_description(job_id: str) -> Optional[Dict]:
    """Get job description by job_id"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM job_descriptions WHERE job_id = ?', (job_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            data = dict(row)
            data['job_requirements'] = json.loads(data['job_requirements'])
            return data
        return None
    except Exception as e:
        print(f"Error getting job description: {e}")
        return None

def save_ranking(user_id: int, job_id: str, ranking_data: List[Dict]) -> bool:
    """Save ranking results"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO rankings (user_id, job_id, ranking_data) VALUES (?, ?, ?)',
            (user_id, job_id, json.dumps(ranking_data))
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving ranking: {e}")
        return False

# Initialize database on import
init_db()

