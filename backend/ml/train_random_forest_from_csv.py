"""
Training script for Random Forest Regressor using data/resume_data.csv

This script:
- Loads the dataset from ../data/resume_data.csv
- Engineers numeric features from `skills` and `skills_required` (and a few others)
- Trains a RandomForestRegressor on `matched_score`
- Saves the trained model to ../models/random_forest_regressor.pkl
"""

import os
import ast
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .models import MLModelTrainer


def _parse_skills(value: str) -> List[str]:
    """
    Parse skills from a string representation.

    The CSV stores lists like "['Python', 'ML']". We try literal_eval first,
    then fall back to a simple comma-split.
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []

    # Try to interpret as Python literal list
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(v).strip() for v in parsed]
    except Exception:
        pass

    # Fallback: split by comma
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return parts


def _extract_experience_years(text: str) -> float:
    """
    Extract approximate required experience (in years) from a free-text field.
    Examples: "At least 3 years", "1-2 years", "Minimum 5 yrs"
    """
    if not isinstance(text, str):
        return 0.0

    # Look for patterns like "3 years", "5 yrs", "1-2 years"
    # Prefer ranges; otherwise take the first number.
    range_match = re.search(r"(\\d+)\\s*[-to]+\\s*(\\d+)\\s*(year|yr)", text, re.IGNORECASE)
    if range_match:
        start = float(range_match.group(1))
        end = float(range_match.group(2))
        return (start + end) / 2.0

    single_match = re.search(r"(\\d+(?:\\.\\d+)?)\\s*(year|yr)", text, re.IGNORECASE)
    if single_match:
        return float(single_match.group(1))

    return 0.0


def _build_features_from_row(row: pd.Series) -> np.ndarray:
    """
    Build a compact numeric feature vector from a CSV row.

    Features include:
    - Number of required skills
    - Number of candidate skills
    - Number of matched skills
    - Match ratios (matched/required, matched/candidate)
    - Required experience (years)
    - Length of job position name (words)
    - Length of candidate career objective (words)
    """
    # Parse skills
    cand_skills = _parse_skills(row.get("skills", ""))  # candidate skills
    req_skills = _parse_skills(row.get("skills_required", ""))  # job required skills

    cand_skills_norm = {s.lower().strip() for s in cand_skills if s}
    req_skills_norm = {s.lower().strip() for s in req_skills if s}

    n_req = float(len(req_skills_norm))
    n_cand = float(len(cand_skills_norm))

    # Count matched skills (using simple normalization + substring check)
    matched = 0
    for req in req_skills_norm:
        for cand in cand_skills_norm:
            if req == cand or req in cand or cand in req:
                matched += 1
                break

    n_matched = float(matched)

    match_ratio_req = n_matched / n_req if n_req > 0 else 0.0
    match_ratio_cand = n_matched / n_cand if n_cand > 0 else 0.0

    # Experience requirement
    exp_text = row.get("experiencere_requirement", "")
    exp_years = _extract_experience_years(exp_text)

    # Text lengths
    job_title = str(row.get("﻿job_position_name", "") or row.get("job_position_name", ""))
    career_obj = str(row.get("career_objective", ""))

    job_title_words = float(len(job_title.split())) if job_title else 0.0
    career_obj_words = float(len(career_obj.split())) if career_obj else 0.0

    features = np.array(
        [
            n_req,
            n_cand,
            n_matched,
            match_ratio_req,
            match_ratio_cand,
            exp_years,
            job_title_words,
            career_obj_words,
        ],
        dtype=float,
    )

    return features


def build_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the raw DataFrame into X (features) and y (matched_score).
    """
    # Drop rows without a valid matched_score
    df = df.copy()
    df = df.dropna(subset=["matched_score"])

    # Ensure matched_score is float
    df["matched_score"] = pd.to_numeric(df["matched_score"], errors="coerce")
    df = df.dropna(subset=["matched_score"])

    feature_list = []
    target_list = []

    for _, row in df.iterrows():
        try:
            features = _build_features_from_row(row)
            if np.any(np.isfinite(features)):
                feature_list.append(features)
                target_list.append(float(row["matched_score"]))
        except Exception:
            # Skip problematic rows but continue training
            continue

    X = np.vstack(feature_list) if feature_list else np.empty((0, 8), dtype=float)
    y = np.array(target_list, dtype=float)
    return X, y


def main():
    # Resolve paths
    # This file lives in backend/ml/, so the project root is two levels up.
    this_dir = os.path.dirname(os.path.abspath(__file__))          # .../backend/ml
    backend_root = os.path.dirname(this_dir)                       # .../backend
    project_root = os.path.dirname(backend_root)                   # .../ (repo root)

    data_path = os.path.join(project_root, "data", "resume_data.csv")
    models_dir = os.path.join(backend_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "random_forest_regressor.pkl")

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)

    print("Building feature matrix...")
    X, y = build_dataset(df)

    if X.shape[0] == 0:
        raise RuntimeError("No valid samples were built from the dataset.")

    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Train/test split for regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trainer = MLModelTrainer()
    print("Training Random Forest Regressor...")
    metrics = trainer.train_random_forest_regressor(
        X_train, y_train, X_test, y_test, n_estimators=200
    )

    print("Training complete. Metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")

    print(f"Saving model to: {model_path}")
    trainer.save_model("rf_regressor", model_path)
    print("Model saved successfully.")


if __name__ == "__main__":
    main()


