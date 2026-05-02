# FastAPI Backend - AI Academic Integrity System
# Serves all 3 modules as API endpoints
# Team - Abhinandan, Stuti, Tushar

import os
import torch
import pickle
import string
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse import hstack
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(
    title="Academic Integrity Risk Detection API",
    description="API for detecting academic integrity risks using AI",
    version="1.0.0"
)

# allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── Database Setup ────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("database/results.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id      TEXT,
            student_name    TEXT,
            module1_score   REAL,
            module2_score   REAL,
            module3_score   REAL,
            composite_score REAL,
            risk_level      TEXT,
            behavior_label  TEXT,
            grade_jump      REAL,
            G1              INTEGER,
            G2              INTEGER,
            G3              INTEGER,
            absences        INTEGER,
            studytime       INTEGER,
            failures        INTEGER,
            analyzed_at     TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized!")

os.makedirs("database", exist_ok=True)
init_db()


# ── Load Models ───────────────────────────────────────────────────────────────
print("Loading all models...")

m1_tokenizer = AutoTokenizer.from_pretrained("Tushar101/module1-roberta")
m1_model     = AutoModelForSequenceClassification.from_pretrained("Tushar101/module1-roberta")
m1_model.eval()

m2_model  = pickle.load(open("models/module2_model.pkl",  "rb"))
m2_scaler = pickle.load(open("models/module2_scaler.pkl", "rb"))
m2_tfidf  = pickle.load(open("models/module2_tfidf.pkl",  "rb"))

m3_model  = pickle.load(open("models/module3_model.pkl",  "rb"))
m3_scaler = pickle.load(open("models/module3_scaler.pkl", "rb"))

print("All models loaded!")


# ── Request Models ────────────────────────────────────────────────────────────
class StudentRequest(BaseModel):
    student_id  : str
    student_name: str
    essay_text  : str
    G1          : int
    G2          : int
    G3          : int
    absences    : int
    studytime   : int
    failures    : int


# ── Prediction Functions ──────────────────────────────────────────────────────
def predict_module1(text):
    inputs = m1_tokenizer(text, return_tensors="pt",
                          truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = m1_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return round(probs[0][1].item() * 100, 2)

def get_writing_features(essay):
    if not isinstance(essay, str) or len(essay.strip()) == 0:
        return [0, 0, 0, 0, 0, 0, 0]
    cleaned = essay.strip()
    raw     = cleaned.split()
    words   = [w.strip(string.punctuation).lower() for w in raw]
    words   = [w for w in words if len(w) > 0]
    for ch in ['!', '?', ';', ':']:
        cleaned = cleaned.replace(ch, '.')
    sentences  = [s.strip() for s in cleaned.split('.') if len(s.strip()) > 5]
    paragraphs = [p.strip() for p in essay.split('\n') if len(p.strip()) > 10]
    linking    = ['however','therefore','moreover','furthermore','although',
                  'nevertheless','consequently','additionally','meanwhile',
                  'otherwise','similarly','thus']
    all_words     = essay.lower().split()
    linking_count = sum(1 for w in linking if w in all_words)
    meaningful    = set('.,!?;:')
    punct_count   = sum(1 for ch in essay if ch in meaningful)
    avg_word_len  = np.mean([len(w) for w in words]) if words else 0
    avg_sent_len  = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    vocab_rich    = len(set(words)) / len(words) if words else 0
    para_count    = len(paragraphs)
    capital_ratio = sum(1 for c in essay if c.isupper()) / len(essay) if essay else 0
    return [avg_word_len, avg_sent_len, vocab_rich,
            punct_count, para_count, linking_count, capital_ratio]

def predict_module2(text):
    features     = get_writing_features(text)
    style_input  = pd.DataFrame([features], columns=[
        'avg_word_len','avg_sent_len','vocab_richness',
        'punct_count','para_count','linking_count','capital_ratio'
    ])
    style_scaled = m2_scaler.transform(style_input)
    tfidf_input  = m2_tfidf.transform([text])
    X_input      = hstack([tfidf_input, sp.csr_matrix(style_scaled)])
    prob         = m2_model.predict_proba(X_input)[0]
    return round(prob[1] * 100, 2)

def predict_module3(G1, G2, G3, absences, studytime, failures):
    baseline          = (G1 + G2) / 2
    grade_jump        = G3 - baseline
    grade_consistency = abs(G1 - G2)
    features          = [[G1, G2, grade_jump, grade_consistency,
                          absences, studytime, failures]]
    scaled            = m3_scaler.transform(features)
    prob              = m3_model.predict_proba(scaled)[0]
    anomaly_prob      = prob[1]
    if grade_jump < 2:
        label = "Normal"
    elif grade_jump >= 7:
        label = "Anomaly"
    elif 3 <= grade_jump < 7:
        label = "Anomaly" if (failures > 0 or absences > 8
                              or anomaly_prob > 0.25) else "Normal"
    else:
        label = "Normal"
    return round(anomaly_prob * 100, 2), label, round(grade_jump, 2)


# ── Save to Database ──────────────────────────────────────────────────────────
def save_to_db(result, request):
    conn   = sqlite3.connect("database/results.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO risk_results (
            student_id, student_name, module1_score, module2_score,
            module3_score, composite_score, risk_level, behavior_label,
            grade_jump, G1, G2, G3, absences, studytime, failures, analyzed_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        result["student_id"], result["student_name"],
        result["module1_ai_score"], result["module2_style_score"],
        result["module3_behavior_score"], result["composite_score"],
        result["risk_level"], result["behavior_label"],
        result["grade_jump"],
        request.G1, request.G2, request.G3,
        request.absences, request.studytime, request.failures,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Academic Integrity Risk Detection API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/history", "/history/{student_id}", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "running", "models": "loaded"}

@app.post("/predict")
def predict(request: StudentRequest):
    try:
        m1 = predict_module1(request.essay_text)
        m2 = predict_module2(request.essay_text)
        m3, beh_label, grade_jump = predict_module3(
            request.G1, request.G2, request.G3,
            request.absences, request.studytime, request.failures
        )

        if beh_label == "Anomaly":
            composite = (0.30 * m1) + (0.30 * m2) + (0.40 * m3)
        else:
            composite = (0.40 * m1) + (0.40 * m2) + (0.20 * m3)

        if composite >= 70:
            risk = "High Risk"
        elif composite >= 55:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        result = {
            "student_id"            : request.student_id,
            "student_name"          : request.student_name,
            "module1_ai_score"      : m1,
            "module2_style_score"   : m2,
            "module3_behavior_score": m3,
            "behavior_label"        : beh_label,
            "composite_score"       : round(composite, 2),
            "risk_level"            : risk,
            "grade_jump"            : grade_jump,
            "analyzed_at"           : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        save_to_db(result, request)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    conn   = sqlite3.connect("database/results.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM risk_results ORDER BY analyzed_at DESC LIMIT 100")
    rows    = cursor.fetchall()
    columns = [d[0] for d in cursor.description]
    conn.close()
    return [dict(zip(columns, row)) for row in rows]

@app.get("/history/{student_id}")
def get_student_history(student_id: str):
    conn   = sqlite3.connect("database/results.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM risk_results WHERE student_id=? ORDER BY analyzed_at DESC",
        (student_id,)
    )
    rows    = cursor.fetchall()
    columns = [d[0] for d in cursor.description]
    conn.close()
    if not rows:
        raise HTTPException(status_code=404, detail="Student not found")
    return [dict(zip(columns, row)) for row in rows]

@app.delete("/history/clear")
def clear_history():
    conn   = sqlite3.connect("database/results.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM risk_results")
    conn.commit()
    conn.close()
    return {"message": "History cleared successfully"}