# Combined Predictor - Module 4
# Integrates all 3 modules and generates composite risk score
# Team - Abhinandan, Stuti, Tushar

import os
import torch
import pickle
import string
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import scipy.sparse as sp
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.system('cls')

# ── Load Module 1 ─────────────────────────────────────────────────────────────
print("Loading Module 1 - RoBERTa...")
m1_tokenizer = AutoTokenizer.from_pretrained("Tushar101/module1-roberta")
m1_model     = AutoModelForSequenceClassification.from_pretrained("Tushar101/module1-roberta")
m1_model.eval()
print("Module 1 loaded!")

# ── Load Module 2 ─────────────────────────────────────────────────────────────
print("Loading Module 2 - Random Forest...")
m2_model  = pickle.load(open("models/module2_model.pkl",  "rb"))
m2_scaler = pickle.load(open("models/module2_scaler.pkl", "rb"))
m2_tfidf  = pickle.load(open("models/module2_tfidf.pkl",  "rb"))
print("Module 2 loaded!")

# ── Load Module 3 ─────────────────────────────────────────────────────────────
print("Loading Module 3 - Behavioral...")
m3_model  = pickle.load(open("models/module3_model.pkl",  "rb"))
m3_scaler = pickle.load(open("models/module3_scaler.pkl", "rb"))
print("Module 3 loaded!")

print("\nAll modules loaded successfully!")

# ── Module 1 Prediction ───────────────────────────────────────────────────────
def predict_module1(text):
    inputs = m1_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = m1_model(**inputs)
    probs   = torch.nn.functional.softmax(outputs.logits, dim=1)
    ai_prob = probs[0][1].item()
    return round(ai_prob * 100, 2)

# ── Module 2 Helper ───────────────────────────────────────────────────────────
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

# ── Module 2 Prediction ───────────────────────────────────────────────────────
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

# ── Module 3 Prediction ───────────────────────────────────────────────────────
def predict_module3(G1, G2, G3, absences, studytime, failures):
    baseline          = (G1 + G2) / 2
    grade_jump        = G3 - baseline
    grade_consistency = abs(G1 - G2)
    features = [[G1, G2, grade_consistency, absences, studytime, failures]]
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
    return round(anomaly_prob * 100, 2), label

# ── Composite Risk Score ──────────────────────────────────────────────────────
def get_composite_risk(student_id, student_name, essay_text,
                        G1, G2, G3, absences, studytime, failures):

    m1_score              = predict_module1(essay_text)
    m2_score              = predict_module2(essay_text)
    m3_score, beh_label   = predict_module3(G1, G2, G3,
                                            absences, studytime, failures)

    # weighted average - text detection weighted more than behavioral
    composite = (0.35 * m1_score) + (0.35 * m2_score) + (0.30 * m3_score)

    if composite >= 65:
        risk = "High Risk"
    elif composite >= 40:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {
        "student_id"            : student_id,
        "student_name"          : student_name,
        "module1_ai_score"      : m1_score,
        "module2_style_score"   : m2_score,
        "module3_behavior_score": m3_score,
        "behavior_label"        : beh_label,
        "composite_score"       : round(composite, 2),
        "risk_level"            : risk
    }

# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n── Test 1: Suspicious Student ──────────────────────")
    result1 = get_composite_risk(
        student_id   = "STU001",
        student_name = "Student A",
        essay_text   = "The implementation of comprehensive educational frameworks necessitates systematic evaluation of pedagogical methodologies to optimize student learning outcomes.",
        G1=8, G2=7, G3=18,
        absences=1, studytime=2, failures=0
    )
    for k, v in result1.items():
        print(f"{k}: {v}")

    print("\n── Test 2: Normal Student ───────────────────────────")
    result2 = get_composite_risk(
        student_id   = "STU002",
        student_name = "Student B",
        essay_text   = "I am a good student with consistent good scores.",
        G1=12, G2=13, G3=14,
        absences=3, studytime=3, failures=0
    )
    for k, v in result2.items():
        print(f"{k}: {v}")