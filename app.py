
import joblib
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load saved model files
model  = joblib.load("models/job_fit_model.pkl")
le     = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/minmax_scaler.pkl")

with open("models/thresholds.json") as f:
    thresholds = json.load(f)

# Job role to required skills mapping
JOB_SKILLS = {
    "Data Scientist":       ["python", "machine learning", "statistics", "sql",
                              "data analysis", "tensorflow", "pandas", "numpy"],
    "Data Analyst":         ["sql", "excel", "python", "tableau", "power bi",
                              "data visualization", "statistics"],
    "Frontend Developer":   ["html", "css", "javascript", "react", "typescript",
                              "responsive design", "git"],
    "Backend Developer":    ["python", "java", "node.js", "sql", "rest api",
                              "docker", "git", "postgresql"],
    "Full Stack Developer": ["html", "css", "javascript", "react", "node.js",
                              "python", "sql", "git", "docker"],
    "Machine Learning Engineer": ["python", "machine learning", "deep learning",
                                   "tensorflow", "pytorch", "mlops", "docker"],
    "DevOps Engineer":      ["docker", "kubernetes", "aws", "ci/cd", "linux",
                              "terraform", "jenkins", "git"],
    "HR Manager":           ["recruitment", "talent acquisition", "hris",
                              "employee relations", "performance management",
                              "onboarding", "payroll"],
    "Software Engineer":    ["python", "java", "data structures", "algorithms",
                              "git", "sql", "rest api", "testing"],
    "Business Analyst":     ["requirements gathering", "sql", "excel",
                              "stakeholder management", "process improvement",
                              "power bi", "documentation"],
    "Cloud Engineer":       ["aws", "azure", "gcp", "docker", "kubernetes",
                              "terraform", "networking", "linux"],
    "Cybersecurity Analyst":["network security", "penetration testing", "siem",
                              "firewalls", "vulnerability assessment",
                              "python", "linux"],
}

def compute_skill_string_score(resume_skills, job_skills):
    resume_set = set(s.lower().strip() for s in resume_skills)
    job_set    = set(s.lower().strip() for s in job_skills)
    if not job_set:
        return 0.0
    matched = resume_set.intersection(job_set)
    return (len(matched) / len(job_set)) * 100

def compute_fuzzy_score(resume_skills, job_skills):
    if not job_skills:
        return 0.0
    scores = []
    for job_skill in job_skills:
        best = max(fuzz.partial_ratio(job_skill.lower(), r.lower())
                   for r in resume_skills) if resume_skills else 0
        scores.append(best)
    return np.mean(scores)

def compute_ai_score(resume_text, job_skills):
    job_text = " ".join(job_skills)
    if not resume_text.strip() or not job_text.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([resume_text, job_text])
        sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        return float(sim * 100)
    except:
        return 0.0

@app.route("/")
def home():
    job_roles = list(JOB_SKILLS.keys())
    return render_template("index.html", job_roles=job_roles)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        resume_skills = [s.strip() for s in data.get("skills", "").split(",") if s.strip()]
        job_role      = data.get("job_role", "")
        experience    = float(data.get("experience", 0))
        education     = data.get("education", "")

        job_skills = JOB_SKILLS.get(job_role, [])
        if not job_skills:
            return jsonify({"error": f"Job role not found: {job_role}"}), 400

        resume_text = " ".join(resume_skills) + " " + education

        ai_score    = compute_ai_score(resume_text, job_skills)
        skill_score = compute_skill_string_score(resume_skills, job_skills)
        fuzzy_score = compute_fuzzy_score(resume_skills, job_skills)

        raw = pd.DataFrame([[ai_score, skill_score, fuzzy_score]],
                            columns=["ai_match_score",
                                     "skill_string_match_score",
                                     "fuzzy_match_score"])
        scaled = scaler.transform(raw)
        ai_n, sk_n, fz_n = scaled[0]

        composite = float(np.clip(ai_n*0.5 + sk_n*0.3 + fz_n*0.2, 0, 1))

        pred  = model.predict(scaled)
        proba = model.predict_proba(scaled)[0]
        label = le.inverse_transform(pred)[0]

        resume_set = set(s.lower().strip() for s in resume_skills)
        job_set    = set(s.lower().strip() for s in job_skills)
        matched    = sorted(resume_set.intersection(job_set))
        missing    = sorted(job_set - resume_set)

        if experience < 1:
            exp_note = "Entry level - suitable for internships or fresher roles"
        elif experience < 3:
            exp_note = "Junior level - good for associate roles"
        elif experience < 6:
            exp_note = "Mid level - suitable for standard positions"
        else:
            exp_note = "Senior level - qualified for lead/senior roles"

        confidence = {cls: round(float(p)*100, 1)
                      for cls, p in zip(le.classes_, proba)}

        return jsonify({
            "prediction":      label,
            "confidence":      confidence,
            "composite_score": round(composite * 100, 1),
            "ai_score":        round(ai_score, 1),
            "skill_score":     round(skill_score, 1),
            "fuzzy_score":     round(fuzzy_score, 1),
            "matched_skills":  matched,
            "missing_skills":  missing,
            "experience_note": exp_note,
            "job_role":        job_role,
            "total_required":  len(job_skills)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
