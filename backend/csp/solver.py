"""
Constraint Satisfaction Problem (CSP) Solver Module
Evaluates candidates against job requirements using constraint satisfaction
"""
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ConstraintType(Enum):
    HARD = "hard"
    SOFT = "soft"

@dataclass
class Constraint:
    """Represents a constraint"""
    name: str
    constraint_type: ConstraintType
    weight: float = 1.0
    satisfied: bool = False
    message: str = ""

class CSPSolver:
    """CSP Solver for candidate evaluation"""
    
    def __init__(self):
        self.variables = {}  # CSP variables
        self.domains = {}    # Variable domains
        self.hard_constraints = []
        self.soft_constraints = []
        self.violations = []
        
    def define_variables(self, job_requirements: Dict):
        """Define CSP variables from job requirements"""
        self.variables = {
            'skills': job_requirements.get('required_skills', []),
            'experience': job_requirements.get('required_experience', 0),
            'degree': job_requirements.get('required_degree', ''),
            'certifications': job_requirements.get('required_certifications', [])
        }
        
        # Define domains
        self.domains = {
            'skills': set(job_requirements.get('required_skills', []) + 
                         job_requirements.get('optional_skills', [])),
            'experience': [0, 1, 2, 3, 4, 5, 10, 15, 20],  # Years
            'degree': ['bachelor', 'master', 'phd', 'none'],
            'certifications': set(job_requirements.get('required_certifications', []) +
                                 job_requirements.get('optional_certifications', []))
        }
    
    def add_hard_constraint(self, constraint_name: str, 
                          constraint_func: callable, 
                          message: str = ""):
        """Add a hard constraint (must be satisfied)"""
        self.hard_constraints.append({
            'name': constraint_name,
            'func': constraint_func,
            'message': message
        })
    
    def add_soft_constraint(self, constraint_name: str,
                          constraint_func: callable,
                          weight: float = 1.0,
                          message: str = ""):
        """Add a soft constraint (preferred but not required)"""
        self.soft_constraints.append({
            'name': constraint_name,
            'func': constraint_func,
            'weight': weight,
            'message': message
        })
    
    def check_mandatory_skills(self, candidate_skills: List[str], 
                             required_skills: List[str]) -> Constraint:
        """Check if candidate has all mandatory skills"""
        if not required_skills:
            return Constraint(
                name="mandatory_skills",
                constraint_type=ConstraintType.HARD,
                satisfied=True,
                message="No mandatory skills required"
            )
        
        candidate_skills_lower = [s.lower().strip() for s in candidate_skills]
        required_skills_lower = [s.lower().strip() for s in required_skills]
        
        missing_skills = []
        for req_skill in required_skills_lower:
            # Try multiple matching strategies
            found = False
            for cand_skill in candidate_skills_lower:
                # Exact match
                if req_skill == cand_skill:
                    found = True
                    break
                # Substring match (skill in candidate skill or vice versa)
                if req_skill in cand_skill or cand_skill in req_skill:
                    found = True
                    break
                # Handle variations (spaces, hyphens, etc.)
                req_variations = [req_skill, req_skill.replace(' ', '-'), req_skill.replace('-', ' '), 
                                req_skill.replace(' ', ''), req_skill.replace('-', '')]
                cand_variations = [cand_skill, cand_skill.replace(' ', '-'), cand_skill.replace('-', ' '),
                                  cand_skill.replace(' ', ''), cand_skill.replace('-', '')]
                if any(rv in cv or cv in rv for rv in req_variations for cv in cand_variations):
                    found = True
                    break
            
            if not found:
                missing_skills.append(req_skill)
        
        satisfied = len(missing_skills) == 0
        message = f"Missing mandatory skills: {', '.join(missing_skills)}" if missing_skills else "All mandatory skills present"
        
        return Constraint(
            name="mandatory_skills",
            constraint_type=ConstraintType.HARD,
            satisfied=satisfied,
            message=message
        )
    
    def check_required_experience(self, candidate_experience: int,
                                 required_experience: int) -> Constraint:
        """Check if candidate meets required experience"""
        satisfied = candidate_experience >= required_experience
        message = f"Experience: {candidate_experience} years (required: {required_experience})"
        
        return Constraint(
            name="required_experience",
            constraint_type=ConstraintType.HARD,
            satisfied=satisfied,
            message=message
        )
    
    def check_required_degree(self, candidate_education: List[Dict],
                              required_degree: str) -> Constraint:
        """Check if candidate has required degree"""
        if not required_degree:
            return Constraint(
                name="required_degree",
                constraint_type=ConstraintType.HARD,
                satisfied=True,
                message="No degree requirement"
            )
        
        required_lower = required_degree.lower()
        has_degree = any(
            required_lower in str(edu).lower() 
            for edu in candidate_education
        )
        
        message = f"Degree requirement: {required_degree} - {'Satisfied' if has_degree else 'Not satisfied'}"
        
        return Constraint(
            name="required_degree",
            constraint_type=ConstraintType.HARD,
            satisfied=has_degree,
            message=message
        )
    
    def check_required_certifications(self, candidate_certs: List[str],
                                      required_certs: List[str]) -> Constraint:
        """Check if candidate has required certifications"""
        if not required_certs:
            return Constraint(
                name="required_certifications",
                constraint_type=ConstraintType.HARD,
                satisfied=True,
                message="No certification requirement"
            )
        
        candidate_certs_lower = [c.lower() for c in candidate_certs]
        required_certs_lower = [c.lower() for c in required_certs]
        
        missing_certs = [c for c in required_certs_lower 
                        if not any(req in cand for cand in candidate_certs_lower 
                                  for req in [c, c.replace(' ', '-')])]
        
        satisfied = len(missing_certs) == 0
        message = f"Missing certifications: {', '.join(missing_certs)}" if missing_certs else "All required certifications present"
        
        return Constraint(
            name="required_certifications",
            constraint_type=ConstraintType.HARD,
            satisfied=satisfied,
            message=message
        )
    
    def check_optional_skills(self, candidate_skills: List[str],
                              optional_skills: List[str]) -> Constraint:
        """Check optional skills (soft constraint)"""
        if not optional_skills:
            return Constraint(
                name="optional_skills",
                constraint_type=ConstraintType.SOFT,
                satisfied=True,
                weight=0.0,
                message="No optional skills specified"
            )
        
        candidate_skills_lower = [s.lower().strip() for s in candidate_skills]
        optional_skills_lower = [s.lower().strip() for s in optional_skills]
        
        matched_skills = []
        for opt_skill in optional_skills_lower:
            # Try multiple matching strategies
            for cand_skill in candidate_skills_lower:
                # Exact match
                if opt_skill == cand_skill:
                    matched_skills.append(opt_skill)
                    break
                # Substring match
                if opt_skill in cand_skill or cand_skill in opt_skill:
                    matched_skills.append(opt_skill)
                    break
                # Handle variations
                opt_variations = [opt_skill, opt_skill.replace(' ', '-'), opt_skill.replace('-', ' '),
                                opt_skill.replace(' ', ''), opt_skill.replace('-', '')]
                cand_variations = [cand_skill, cand_skill.replace(' ', '-'), cand_skill.replace('-', ' '),
                                 cand_skill.replace(' ', ''), cand_skill.replace('-', '')]
                if any(ov in cv or cv in ov for ov in opt_variations for cv in cand_variations):
                    matched_skills.append(opt_skill)
                    break
        
        match_ratio = len(matched_skills) / len(optional_skills) if optional_skills else 0.0
        
        return Constraint(
            name="optional_skills",
            constraint_type=ConstraintType.SOFT,
            satisfied=match_ratio > 0,
            weight=match_ratio,
            message=f"Matched {len(matched_skills)}/{len(optional_skills)} optional skills"
        )
    
    def evaluate_candidate(self, candidate_data: Dict, 
                          job_requirements: Dict) -> Dict:
        """Evaluate candidate against job requirements using CSP"""
        self.define_variables(job_requirements)
        
        # Extract candidate information
        candidate_skills = candidate_data.get('skills', [])
        candidate_experience = candidate_data.get('experience', {}).get('years', 0)
        candidate_education = candidate_data.get('education', [])
        candidate_certs = candidate_data.get('certifications', [])
        
        # Check hard constraints
        constraints = []
        
        # Mandatory skills
        mandatory_skills = job_requirements.get('required_skills', [])
        skill_constraint = self.check_mandatory_skills(candidate_skills, mandatory_skills)
        constraints.append(skill_constraint)
        
        # Required experience
        required_exp = job_requirements.get('required_experience', 0)
        exp_constraint = self.check_required_experience(candidate_experience, required_exp)
        constraints.append(exp_constraint)
        
        # Required degree
        required_degree = job_requirements.get('required_degree', '')
        degree_constraint = self.check_required_degree(candidate_education, required_degree)
        constraints.append(degree_constraint)
        
        # Required certifications
        required_certs = job_requirements.get('required_certifications', [])
        cert_constraint = self.check_required_certifications(candidate_certs, required_certs)
        constraints.append(cert_constraint)
        
        # Check soft constraints
        optional_skills = job_requirements.get('optional_skills', [])
        optional_skill_constraint = self.check_optional_skills(candidate_skills, optional_skills)
        constraints.append(optional_skill_constraint)
        
        # Calculate scores
        hard_constraints_satisfied = all(
            c.satisfied for c in constraints 
            if c.constraint_type == ConstraintType.HARD
        )
        
        soft_constraints_score = sum(
            c.weight for c in constraints 
            if c.constraint_type == ConstraintType.SOFT
        )
        
        # Eligibility score (0-1)
        if not hard_constraints_satisfied:
            eligibility_score = 0.0
        else:
            # Check if there are any required constraints at all
            has_required_skills = len(mandatory_skills) > 0
            has_required_exp = required_exp > 0
            has_required_degree = bool(required_degree)
            has_required_certs = len(required_certs) > 0
            
            # If no hard constraints exist (all optional), base score should be lower
            # and depend on whether candidate has any skills at all
            if not has_required_skills and not has_required_exp and not has_required_degree and not has_required_certs:
                # No hard constraints - score based on optional skills and candidate having skills
                if len(candidate_skills) == 0:
                    # No skills at all - very low score
                    eligibility_score = 0.1
                else:
                    # Has some skills - score based on optional skill match
                    base_score = 0.3  # Lower base when no hard constraints
                    soft_bonus = min(soft_constraints_score * 0.4, 0.4)
                    eligibility_score = base_score + soft_bonus
            else:
                # Has hard constraints and they're satisfied
                base_score = 0.7
                # Bonus from soft constraints
                soft_bonus = min(soft_constraints_score * 0.3, 0.3)
                eligibility_score = base_score + soft_bonus
        
        # Collect violations
        violations = [
            {
                'constraint': c.name,
                'type': c.constraint_type.value,
                'message': c.message,
                'satisfied': c.satisfied
            }
            for c in constraints if not c.satisfied
        ]
        
        # Missing skills
        missing_skills = []
        if not skill_constraint.satisfied:
            candidate_skills_lower = [s.lower() for s in candidate_skills]
            for req_skill in mandatory_skills:
                req_skill_lower = req_skill.lower()
                if not any(req_skill_lower in cand for cand in candidate_skills_lower):
                    missing_skills.append(req_skill)
        
        return {
            'eligibility_score': eligibility_score,
            'hard_constraints_satisfied': hard_constraints_satisfied,
            'soft_constraints_score': soft_constraints_score,
            'constraints': [
                {
                    'name': c.name,
                    'type': c.constraint_type.value,
                    'satisfied': c.satisfied,
                    'message': c.message,
                    'weight': getattr(c, 'weight', 1.0)
                }
                for c in constraints
            ],
            'violations': violations,
            'missing_skills': missing_skills
        }





