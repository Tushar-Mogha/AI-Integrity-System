# AI-Assisted Academic Integrity Risk Detection System
# Streamlit Dashboard
# Team - Abhinandan, Stuti, Tushar

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import requests
import pickle
import string
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.sparse import hstack
import scipy.sparse as sp
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from modules.module4_explainability import generate_shap_plot, get_text_explanation
from app.database import save_result, get_all_results, get_student_history, update_note, delete_record
from datetime import datetime, timedelta, timezone
from app.database import get_all_results
from modules.module5_plagiarism import check_plagiarism


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Academic Integrity Risk Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main { background-color: #0F1923; }

    .stApp {
        background: linear-gradient(135deg, #0F1923 0%, 1A2634 100%);
    }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        color: #E8E0D0 !important;
    }

    .header-container {
        background: linear-gradient(135deg, #1B2A4A 0%, #0F1923 100%);
        border-left: 4px solid #C9A84C;
        padding: 2rem 2.5rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 2rem;
    }

    .header-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #E8E0D0;
        margin: 0;
        line-height: 1.2;
    }

    .header-subtitle {
        font-family: 'DM Sans', sans-serif;
        color: #8A99B0;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1B2A4A, #162236);
        border: 1px solid #2A3F5F;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2.5rem;
        color: #C9A84C;
        margin: 0;
    }

    .metric-label {
        font-family: 'DM Sans', sans-serif;
        color: #8A99B0;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    .risk-high {
        background: linear-gradient(135deg, #4A1B1B, #3D1515);
        border: 1px solid #8B3A3A;
        border-radius: 8px;
        padding: 0.4rem 1rem;
        color: #FF6B6B;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .risk-medium {
        background: linear-gradient(135deg, #4A3A1B, #3D2E15);
        border: 1px solid #8B6A3A;
        border-radius: 8px;
        padding: 0.4rem 1rem;
        color: #FFB347;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .risk-low {
        background: linear-gradient(135deg, #1B4A2A, #153D22);
        border: 1px solid #3A8B5A;
        border-radius: 8px;
        padding: 0.4rem 1rem;
        color: #6BCB77;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .student-card {
        background: linear-gradient(135deg, #1B2A4A, #162236);
        border: 1px solid #2A3F5F;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: border-color 0.2s;
    }

    .student-card:hover {
        border-color: #C9A84C;
    }

    .section-header {
        font-family: 'DM Serif Display', serif;
        color: #C9A84C;
        font-size: 1.1rem;
        border-bottom: 1px solid #2A3F5F;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    .info {
        color: #8A99B0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    # Module 1
    m1_tokenizer = AutoTokenizer.from_pretrained("Tushar101/module1-roberta")
    m1_model     = AutoModelForSequenceClassification.from_pretrained("Tushar101/module1-roberta")
    m1_model.eval()

    # Module 2
    m2_model  = pickle.load(open("models/module2_model.pkl",  "rb"))
    m2_scaler = pickle.load(open("models/module2_scaler.pkl", "rb"))
    m2_tfidf  = pickle.load(open("models/module2_tfidf.pkl",  "rb"))

    # Module 3
    m3_model  = pickle.load(open("models/module3_model.pkl",  "rb"))
    m3_scaler = pickle.load(open("models/module3_scaler.pkl", "rb"))

    return m1_tokenizer, m1_model, m2_model, m2_scaler, m2_tfidf, m3_model, m3_scaler

m1_tokenizer, m1_model, m2_model, m2_scaler, m2_tfidf, m3_model, m3_scaler = load_models()


# ── Prediction Functions ──────────────────────────────────────────────────────
def predict_module1(text):
    inputs = m1_tokenizer(text, return_tensors="pt", truncation=True,
                          padding=True, max_length=512)
    with torch.no_grad():
        outputs = m1_model(**inputs)
    probs   = torch.nn.functional.softmax(outputs.logits, dim=1)
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
    ai_prob      = round(prob[1] * 100, 2)
    label        = "AI Generated" if ai_prob >= 60 else "Human Written"
    return ai_prob, label

def predict_module3(G1, G2, G3, absences, studytime, failures):
    baseline = (G1 + G2) / 2
    grade_jump = G3 - baseline
    jump_abs = abs(grade_jump)
    trend = G2 - G1
    consistency = abs(G1 - G2)
    avg_score = (G1 + G2) / 2
    features = [[
        G1, G2,
        trend,
        consistency,
        avg_score,
        absences,
        studytime,
        failures
    ]]

    scaled = m3_scaler.transform(features)
    prob = m3_model.predict_proba(scaled)[0]
    model_anomaly_prob = prob[1]

    # -----------------------------
    # RULE LOGIC
    # -----------------------------
    if grade_jump >= 6 or grade_jump <= -4:
        label = "Anomaly"
        final_prob = max(model_anomaly_prob, 0.90)
    
    elif consistency >= 5 and jump_abs >= 4:
        label = "Anomaly"
        final_prob = max(model_anomaly_prob, 0.75)

    elif jump_abs >= 3.5 and consistency <= 1:
        label = "Anomaly"
        final_prob = max(model_anomaly_prob, 0.70)

    elif absences > 10 or failures > 1:
        label = "Anomaly"
        final_prob = max(model_anomaly_prob, 0.75)

    # elif consistency <= 2 and 3 < jump_abs < 3.5:
    #     label = "Anomaly"
    #     final_prob = max(model_anomaly_prob, 0.60)

    else:
        label = "Normal"
        final_prob = min(model_anomaly_prob, 0.30)

    # -----------------------------
    # REASON 
    # -----------------------------
    if label == "Anomaly":

        if grade_jump > 6:
            reason = "Extreme increase in performance detected"

        elif grade_jump <= -4:
            reason = "Extreme decrease in performance detected"
        
        elif consistency >= 5 and jump_abs >= 4:
            reason = "High fluctuation in performance detected"

        elif abs(grade_jump) >= 3.5:
            reason = "Significant deviation from baseline"

        elif absences > 10:
            reason = "High absences with unusual performance"

        elif failures > 1:
            reason = "Failures with inconsistent performance"

        else:
            reason = "Unusual performance pattern detected"

    else:
        reason = "Performance within expected range"

    final_prob = round(final_prob * 100, 2)

    return final_prob, label, round(grade_jump, 2), reason

def get_risk(composite):
    if composite >= 70:   return "High Risk"
    elif composite >= 55: return "Medium Risk"
    else:                 return "Low Risk"

def risk_color(risk):
    return {"High Risk":"#FF6B6B","Medium Risk":"#FFB347","Low Risk":"#6BCB77"}.get(risk,"#8A99B0")

def analyze_student(sid, name, essay, G1, G2, G3, absences, studytime, failures):
    m1 = predict_module1(essay)
    m2, m2_label = predict_module2(essay)
    m3, beh_label, grade_jump, m3_reason = predict_module3(
        G1, G2, G3, absences, studytime, failures)
    if beh_label == "Anomaly":
        composite = (0.35 * m1) + (0.35 * m2) + (0.30 * m3)
    else:
        composite = (0.30 * m1) + (0.30 * m2) + (0.40 * m3)
    wf   = get_writing_features(essay)

    past_records = get_all_results()

    plagiarism = check_plagiarism(essay, past_records)

    copy_flag = plagiarism["is_copied"]
    copy_score = plagiarism["similarity"]
    matched_student = plagiarism["matched_student"]

    # ── Composite Score Adjustment using Module 5 ───────

    # Severe plagiarism
    if copy_score >= 80:
        composite += 45 

    # Moderate plagiarism
    elif copy_score >= 60:
        composite += 30

    # Mild plagiarism
    elif copy_score >= 40:
        composite += 15

    # Cap score at 100
    composite = min(composite, 100)

    risk = get_risk(composite)

    result= {
        "student_id"            : sid,
        "student_name"          : name,
        "essay_text"            : essay,
        "copy_detected"         : copy_flag,
        "copy_score"            : copy_score,
        "matched_student"       : matched_student,
        "module1_ai_score"      : m1,
        "module2_style_score"   : m2,
        "module2_label"         : m2_label,
        "module3_behavior_score": m3,
        "behavior_label"        : beh_label,
        "module3_reason"        : m3_reason,
        "composite_score"       : round(composite, 2),
        "risk_level"            : risk,
        "grade_jump"            : grade_jump,
        "baseline"              : round((G1+G2)/2, 2),
        "G1": G1, "G2": G2, "G3": G3,
        "absences" : absences,
        "studytime": studytime,
        "failures" : failures,
        "writing_features": {
            "Avg Word Length"   : wf[0],
            "Avg Sentence Length": wf[1],
            "Vocabulary Richness": wf[2],
            "Punctuation Count" : wf[3],
            "Paragraph Count"   : wf[4],
            "Linking Words"     : wf[5]
        }
    }
    # save to database and store record id in session
    try:
        record_id = save_result(result)
        if record_id:
            st.session_state[f"record_id_{sid}"] = record_id
            print(f"Record ID stored in session: {record_id}")
    except Exception as e:
        print(f"Database Error: {str(e)}")

    return result

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0; border-bottom:1px solid #2A3F5F; margin-bottom:1.5rem;'>
        <div style='font-family:DM Serif Display,serif; font-size:1.3rem; color:#C9A84C;'>
        🛡️ AcademicGuard
        </div>
        <div style='font-size:0.7rem; color:#8A99B0; letter-spacing:1px;'>
        AI INTEGRITY SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
    "Navigation",
    ["🏠  Home", "👤  Individual Check", "📊  Class Analysis", "📜  History"],
    label_visibility="collapsed"
    )

    st.markdown("""
    <div style='margin-top:1rem;
                font-size:0.75rem; color:#8A99B0;'>
    <b style='color:#C9A84C; font-size:0.8rem;'>Risk Levels</b><br>
    <span style='color:#FF6B6B;'>🔴 High Risk</span> ≥ 70<br>
    <span style='color:#FFB347;'>🟡 Medium Risk</span> ≥ 55<br>
    <span style='color:#6BCB77;'>🟢 Low Risk</span> &lt; 55
    
    <br><br>
    <b style='color:#C9A84C;'>System Modules</b><br>
    🧠 M1 · RoBERTa AI Detection<br>
    ✍️ M2 · Writing Style Analysis<br>
    📈 M3 · Behavioral Anomaly<br>
    🔍 M4 · Explainable AI (SHAP)<br>
    📄 M5 · Peer Plagiarism Detection

    <br><br>
    <b style='color:#C9A84C;'>Team</b><br>
    Abhinandan Kumar<br>
    Tushar Mogha<br>
    Stuti Mishra<br>
    UPES · SoCS · 2026
    </div>
    """, unsafe_allow_html=True)


# ── Helper to show student report ─────────────────────────────────────────────
def show_student_report(result):
    rc = risk_color(result["risk_level"])

    # Header
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        st.markdown(f"""
        <div style='background:#1B2A4A; border:1px solid #2A3F5F; border-radius:12px;
                    padding:1.2rem;'>
            <div style='font-size:0.75rem; color:#8A99B0; letter-spacing:1px;
                        text-transform:uppercase;'>Student</div>
            <div style='font-family:DM Serif Display,serif; font-size:1.6rem;
                        color:#E8E0D0;'>{result['student_name']}</div>
            <div style='color:#8A99B0; font-size:0.85rem;'>ID: {result['student_id']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:{rc};'>{result['composite_score']}</div>
            <div class='metric-label'>Composite Score</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        emoji = {"High Risk":"🔴","Medium Risk":"🟡","Low Risk":"🟢"}.get(result['risk_level'],"⚪")
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:{rc}; font-size:1.4rem;'>
            {emoji}<br>{result['risk_level']}</div>
            <div class='metric-label'>Risk Level</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge charts
    st.markdown("<div class='section-header'>Module Score Breakdown</div>",
                unsafe_allow_html=True)
    gc1, gc2, gc3 = st.columns(3)
    for col, mod, score, color, extra_label,extra_type in zip(
        [gc1, gc2, gc3],
        ["Module 1 · AI Detection", "Module 2 · Writing Style", "Module 3 · Behavioral"],
        [result["module1_ai_score"], result["module2_style_score"], result["module3_behavior_score"]],
        ["#3498DB", "#C9A84C", "#27AE60"],
        ["", result.get("module2_label"), ""],
        ["copy", "style", ""]
    ):
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix":"%","font":{"size":28,"color":color}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#8A99B0","tickfont":{"size":10}},
                    "bar":{"color":color},
                    "bgcolor":"rgba(27,42,74,0.8)",
                    "bordercolor":"#2A3F5F",
                    "steps":[
                        {"range":[0,40],"color":"rgba(39,174,96,0.08)"},
                        {"range":[40,70],"color":"rgba(243,156,18,0.08)"},
                        {"range":[70,100],"color":"rgba(231,76,60,0.08)"},
                    ],
                    "threshold":{"line":{"color":"#E74C3C","width":2},
                                 "thickness":0.75,"value":70}
                },
                title={"text":f"<b>{mod}</b>","font":{"size":12,"color":"#E8E0D0"}}
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=10,r=10,t=30,b=10), height=220,
                              font=dict(color="#E8E0D0"))
            st.plotly_chart(fig, use_container_width=True)

            # --- MODULE 1 COPY BOX ---
            if extra_type == "copy":
    
                copy_score = result.get("copy_score", 0)
                matched_student = result.get("matched_student", "")

                # Get student name instead of ID
                matched_name = ""
                if matched_student:
                    try:
                        from app.database import get_all_results
                        all_records = get_all_results()
                        for r in all_records:
                            if str(r["student_id"]) == str(matched_student):
                                matched_name = r["student_name"]
                                break
                    except:
                        matched_name = matched_student

                # UI Rendering
                if copy_score > 0:
                    st.markdown(f"""
                    <div style='text-align:center; margin-top:-1rem; margin-bottom:0.6rem;'>
                        <div style='background:#1E2A44; border:1px solid #E74C3C;
                                    border-radius:10px; padding:0.6rem; font-size:0.9rem;'>
                            <span style='color:#E74C3C; font-weight:600;'>
                                📄 Possible Copy: {copy_score}%
                            </span><br>
                            <span style='color:#E8E0D0;'>
                                Match: {matched_name if matched_name else "N/A"}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='text-align:center; margin-top:-1rem; margin-bottom:0.6rem;'>
                        <div style='background:#1E2A44; border:1px solid #2ECC71;
                                    border-radius:10px; padding:0.6rem; font-size:0.9rem;'>
                            <span style='color:#2ECC71; font-weight:600;'>
                                📄 Possible Copy: 0%
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # show Module 2 label as colored badge below gauge
            if extra_label:
                badge_color = "#E74C3C" if extra_label == "AI Generated" else "#27AE60"
                badge_icon  = "🤖" if extra_label == "AI Generated" else "✍️"
                st.markdown(f"""
                <div style='text-align:center; margin-top:-1rem; margin-bottom:0.5rem;'>
                    <span style='background:{badge_color}22; border:1px solid {badge_color};
                                 color:{badge_color}; border-radius:20px;
                                 padding:0.4rem 1.2rem; font-size:1rem;
                                 font-weight:700; letter-spacing:0.5px;'>
                        {badge_icon} {extra_label}
                    </span>
                </div>
                """, unsafe_allow_html=True)

    # Grade chart + behavioral summary
    st.markdown("<div class='section-header'>Grade Pattern Analysis</div>",
                unsafe_allow_html=True)
    pc1, pc2 = st.columns([2,1])
    with pc1:
        anomaly = result["behavior_label"] == "Anomaly"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=["G1 (Period 1)","G2 (Period 2)","G3 (Final)"],
            y=[result["G1"], result["G2"], result["G3"]],
            mode="lines+markers+text",
            text=[result["G1"], result["G2"], result["G3"]],
            textposition="top center",
            line=dict(color="#C9A84C", width=3),
            marker=dict(
                size=12,
                color=["#3498DB","#3498DB","#E74C3C" if anomaly else "#27AE60"],
                line=dict(width=2, color="#E8E0D0")
            ),
            fill="tozeroy",
            fillcolor="rgba(201,168,76,0.06)"
        ))
        fig.add_hline(y=result["baseline"], line_dash="dash", line_color="#8A99B0",
                      annotation_text=f"Baseline: {result['baseline']}",
                      annotation_font_color="#8A99B0")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(27,42,74,0.4)",
                          margin=dict(l=10,r=10,t=20,b=10), height=260,
                          font=dict(color="#E8E0D0"),
                          yaxis=dict(range=[0,21],gridcolor="rgba(42,63,95,0.5)"),
                          xaxis=dict(gridcolor="rgba(42,63,95,0.5)"),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with pc2:
        jc = "#E74C3C" if result["grade_jump"] > 5 else "#27AE60"
        bc = "#E74C3C" if anomaly else "#27AE60"
        if result["behavior_label"] == "Anomaly":
            reason_color = "#FF6B6B"
        else:
            reason_color = "#6BCB77"
        st.markdown(f"""
        <div style='background:#1B2A4A; border:1px solid #2A3F5F; border-left:3px solid #C9A84C;
                    border-radius:12px; padding:1.2rem; height:100%;'>
        <div style='font-size:0.8rem; color:#8A99B0;'>Baseline Score</div>
        <div style='font-size:1.3rem; font-weight:700; color:#E8E0D0;'>{result['baseline']}</div>
        <br>
        <div style='font-size:0.8rem; color:#8A99B0;'>Grade Jump</div>
        <div style='font-size:1.3rem; font-weight:700; color:{jc};'>{result['grade_jump']}</div>
        <br>
        <div style='font-size:0.8rem; color:#8A99B0;'>Behavior Label</div>
        <div style='font-weight:700; color:{bc};'>{result['behavior_label']}</div>
        <div style='font-size:0.8rem; color:#8A99B0; margin-top:0.4rem;'>Reason</div>
        <div style='color:{reason_color}; font-size:0.85rem; font-weight:600;'>
        {result.get("module3_reason", "N/A")}
        </div>
        <br>
        <div style='font-size:0.8rem; color:#8A99B0;'>
        Absences: <b style='color:#E8E0D0;'>{result['absences']}</b> &nbsp;|&nbsp;
        Failures: <b style='color:#E8E0D0;'>{result['failures']}</b>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Writing features bar chart
    st.markdown("<div class='section-header'>Writing Style Features</div>",
                unsafe_allow_html=True)
    wf  = result["writing_features"]
    fig = go.Figure(go.Bar(
        x=list(wf.keys()),
        y=list(wf.values()),
        marker=dict(
            color=["#3498DB","#C9A84C","#27AE60","#E74C3C","#9B59B6","#1ABC9C"],
            line=dict(color="rgba(42,63,95,0.5)", width=1)
        ),
        text=[f"{v:.2f}" for v in wf.values()],
        textposition="outside",
        textfont=dict(color="#E8E0D0", size=11)
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(27,42,74,0.4)",
                      margin=dict(l=10,r=10,t=20,b=10), height=250,
                      font=dict(color="#E8E0D0"),
                      yaxis=dict(gridcolor="rgba(42,63,95,0.5)"),
                      xaxis=dict(gridcolor="rgba(42,63,95,0.5)"))
    st.plotly_chart(fig, use_container_width=True)

    # SHAP Explanation
    st.markdown("<div class='section-header'>SHAP Writing Style Analysis</div>",
                unsafe_allow_html=True)

    essay_text = result.get("essay_text", "")
    if essay_text:
        try:
            import sys
            sys.path.append(".")
            from modules.module4_explainability import generate_shap_plot, get_text_explanation
            import os

            os.makedirs("outputs", exist_ok=True)
            shap_path = f"outputs/shap_{result['student_id']}.png"

            # always regenerate
            with st.spinner("Generating SHAP explanation..."):
                generate_shap_plot(
                    essay        = essay_text,
                    student_name = result["student_name"],
                    save_path    = shap_path
                )

            if os.path.exists(shap_path):
                st.image(shap_path, width=700)
            else:
                st.warning("SHAP plot could not be generated.")

            st.markdown("<b style='color:#C9A84C;'>Feature Impact Explanation:</b>",
                        unsafe_allow_html=True)
            explanations = get_text_explanation(
                essay_text,
                result["module2_style_score"]
            )
            for exp in explanations:
                st.markdown(f"""
                <div style='background:#1B2A4A; border-left:3px solid #C9A84C;
                            border-radius:6px; padding:0.6rem 1rem; margin:0.3rem 0;
                            font-size:0.85rem; color:#8A99B0;'>
                {exp}
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"SHAP analysis unavailable: {str(e)}")
    else:
        st.info("Essay text not available for SHAP analysis")

    # Risk explanation
    st.markdown("<div class='section-header'>Risk Explanation</div>",
                unsafe_allow_html=True)
    reasons = []
    if result.get("copy_detected"):
        reasons.append( f"Essay shows {result['copy_score']}% similarity with another student submission" )
    if result["module1_ai_score"] > 70:
        reasons.append(f"AI Detection model flagged essay with {result['module1_ai_score']}% AI probability")
    if result["module2_style_score"] > 70:
        reasons.append(f"Writing style shows AI-like patterns with {result['module2_style_score']}% confidence")
    if result["behavior_label"] == "Anomaly":
        reasons.append(f"Behavioral anomaly detected: {result['module3_reason']}")
    if result["failures"] > 0:
        reasons.append(f"Student has {result['failures']} past failure(s) — high final grade is suspicious")
    if result["absences"] > 8:
        reasons.append(f"High absences ({result['absences']}) combined with improved performance is unusual")
    if not reasons:
        reasons.append("No strong indicators of academic misconduct detected")

    items = "".join([f"<li style='margin:0.4rem 0; color:#8A99B0;'>{r}</li>"
                     for r in reasons])
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(201,168,76,0.06),rgba(15,25,35,0.5));
                border:1px solid #2A3F5F; border-left:3px solid #C9A84C;
                border-radius:12px; padding:1.2rem;'>
    <b style='color:#C9A84C;'>Why was this student flagged?</b>
    <ul style='margin-top:0.8rem; padding-left:1.2rem;'>{items}</ul>
    <div style='margin-top:0.8rem; font-size:0.8rem; color:#8A99B0;
                border-top:1px solid #2A3F5F; padding-top:0.8rem;'>
    This is a decision-support tool. Final decisions rest with the faculty member.
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Faculty notes
    st.markdown("<div class='section-header'>Faculty Notes</div>",
                unsafe_allow_html=True)

    note = st.text_area(
        "Add observations", height=100,
        placeholder="Enter any additional observations...",
        key=f"note_{result['student_id']}"
    )

    if st.button("Save Note", key=f"save_btn_{result['student_id']}"):
        if note.strip():
            try:
                history = get_student_history(result["student_id"])
            
                if history:
                    record_id = history[0]["id"]
                    success = update_note(record_id, note)
                    if success:
                        st.success("Note saved! Go to History page.")
                        st.balloons()
                    else:
                        st.warning("Note could not be saved. Try again.")
                else:
                    st.warning("No record found. Analyze the student first.")

            except Exception as e:
                st.error(f"Could not save note: {str(e)}")
        else:
            st.warning("Please write something before saving.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if "🏠" in page:
    st.markdown("""
    <div class='header-container'>
        <div class='header-title'>AI-Assisted Academic<br>Integrity Risk Detection</div>
        <div class='header-subtitle'>
        Detect AI-generated content · Analyze writing style · Flag behavioral anomalies · Detect peer plagiarism
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, val, lbl in zip(
    [c1,c2,c3,c4],
    ["99.60%","98.17%","91.96%","5"],   
    ["M1 Accuracy","M2 Accuracy","M3 Accuracy","System Modules"]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>How It Works</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#1B2A4A;
                border:1px solid #2A3F5F; border-left:3px solid #9B59B6;
                border-radius:12px; padding:1rem 1.2rem; margin-bottom:1rem;'>

    <b style='color:#9B59B6;'>🔍 Explainable AI (XAI)</b><br>

    <span style='color:#8A99B0; font-size:0.85rem;'>
    SHAP-based explainability highlights which writing features contributed most to the AI-writing prediction, improving transparency and faculty trust.
    </span>

    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    modules_info = [
        ("🤖","Module 1","AI Text Detection",
         "Fine-tuned RoBERTa detects AI-generated text with 99.60% accuracy on 44,868 essays","#3498DB"),
        ("✍️","Module 2","Writing Style Analysis",
         "TF-IDF + Random Forest extracts 7 linguistic features to detect AI writing with 98.17% accuracy","#C9A84C"),
        ("📈","Module 3","Behavioral Anomaly",
         "Detects suspicious grade jumps using Random Forest on UCI Student Performance data","#27AE60"),
        ("📄","Module 5","Peer Plagiarism Detection",
        "Uses TF-IDF cosine similarity to detect copied submissions between students","#E74C3C"),
    ]
    for col, (icon,mod,title,desc,color) in zip([c1,c2,c3,c4], modules_info):
        with col:
            st.markdown(f"""
            <div style='background:#1B2A4A; border:1px solid #2A3F5F;
                        border-top:3px solid {color}; border-radius:12px;
                        padding:1.5rem; height:260px; display:flex; flex-direction:column; justify-content:space-between;'>
                <div style='font-size:1.8rem;'>{icon}</div>
                <div style='font-size:0.7rem; color:#8A99B0; letter-spacing:1px;
                            text-transform:uppercase; margin-top:0.5rem;'>{mod}</div>
                <div style='font-weight:600; color:{color}; margin:0.3rem 0;'>{title}</div>
                <div style='font-size:0.82rem;
                            color:#8A99B0;line-height:1.7;
                            margin-top:0.7rem;
                            overflow-wrap:break-word;'>
                {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>CSV Format Required</div>",
                unsafe_allow_html=True)

    sample = pd.DataFrame({
        "student_id":["STU001","STU002"],
        "student_name":["Rahul Sharma","Priya Singh"],
        "essay_text":["Essay text here...","Essay text here..."],
        "G1":[8,12],"G2":[7,13],"G3":[18,14],
        "absences":[1,3],"studytime":[2,3],"failures":[0,0]
    })
    st.dataframe(sample, use_container_width=True, hide_index=True)

    import io
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    st.download_button("⬇️ Download Sample CSV", buf.getvalue(),
                       "sample_students.csv", "text/csv")

    st.markdown("""
    <div style='background:rgba(201,168,76,0.06); border:1px solid #2A3F5F;
                border-left:3px solid #C9A84C; border-radius:12px;
                padding:1rem 1.2rem; margin-top:1rem;'>
    <b style='color:#C9A84C;'>⚠️ Important</b><br>
    <span style='color:#8A99B0; font-size:0.85rem;'>
    This system combines AI detection, behavioral analytics, peer plagiarism detection, and explainable AI to assist faculty in identifying potential academic integrity risks.
    Risk flags are for faculty review only —  not automated accusations. All final decisions remain with the faculty member.
    </span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INDIVIDUAL CHECK
# ══════════════════════════════════════════════════════════════════════════════
elif "👤" in page:
    st.markdown("<h2>Individual Student Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p class='info'>Enter student details for instant risk assessment</p>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sid  = st.text_input("Student ID",   placeholder="e.g. STU001")
        name = st.text_input("Student Name", placeholder="e.g. Rahul Sharma")
    with c2:
        gc1,gc2,gc3 = st.columns(3)
        G1 = gc1.number_input("G1", 0, 20, 10)
        G2 = gc2.number_input("G2", 0, 20, 10)
        G3 = gc3.number_input("G3", 0, 20, 10)
        gc4,gc5,gc6 = st.columns(3)
        absences  = gc4.number_input("Absences",  0, 100, 3)
        studytime = gc5.number_input("Study Time", 1, 4, 2)
        failures  = gc6.number_input("Failures",   0, 10, 0)

    essay = st.text_area("Essay Text", height=180,
                         placeholder="Paste the student essay here...")

    if st.button("🔍  Analyze Student"):
        if not sid or not name or not essay:
            st.error("Please fill in Student ID, Name, and Essay Text.")
        else:
            with st.spinner("Running analysis across all 3 modules..."):
                result = analyze_student(sid, name, essay, G1, G2, G3,
                                         absences, studytime, failures)
            st.session_state["last_result"] = result

    # show report if result exists in session
    if "last_result" in st.session_state:
        st.markdown("---")
        show_student_report(st.session_state["last_result"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLASS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "📊" in page:
    st.markdown("<h2>Class-Wide Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p class='info'>Upload CSV to analyze entire class at once</p>",
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Student CSV", type=["csv"])

    if uploaded:
        df  = pd.read_csv(uploaded)
        req = ["student_id","student_name","essay_text","G1","G2","G3",
               "absences","studytime","failures"]
        missing = [c for c in req if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            st.success(f"✅ {len(df)} students loaded!")

            if st.button("🚀  Run Analysis on All Students"):
                results = []
                bar = st.progress(0, text="Analyzing...")
                for i, row in df.iterrows():
                    r = analyze_student(
                        str(row["student_id"]), str(row["student_name"]),
                        str(row["essay_text"]),
                        int(row["G1"]), int(row["G2"]), int(row["G3"]),
                        int(row["absences"]), int(row["studytime"]), int(row["failures"])
                    )
                    results.append(r)
                    bar.progress((i+1)/len(df), text=f"Analyzing {row['student_name']}...")
                bar.empty()
                st.session_state["results"] = results
                st.session_state["analyzed"] = True

    if st.session_state.get("analyzed"):
        results = st.session_state["results"]
        rdf     = pd.DataFrame(results)

        # Summary
        total  = len(rdf)
        high   = len(rdf[rdf["risk_level"]=="High Risk"])
        medium = len(rdf[rdf["risk_level"]=="Medium Risk"])
        low    = len(rdf[rdf["risk_level"]=="Low Risk"])

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        for col, val, lbl, color in zip(
            [c1,c2,c3,c4],
            [total, high, medium, low],
            ["Total Students","High Risk","Medium Risk","Low Risk"],
            ["#C9A84C","#FF6B6B","#FFB347","#6BCB77"]
        ):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value' style='color:{color};'>{val}</div>
                    <div class='metric-label'>{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        ch1, ch2 = st.columns(2)
        layout = dict(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(27,42,74,0.4)",
                      margin=dict(l=10,r=10,t=30,b=10),
                      font=dict(color="#E8E0D0"))

        with ch1:
            st.markdown("<div class='section-header'>Risk Distribution</div>",
                        unsafe_allow_html=True)
            fig = go.Figure(go.Pie(
                labels=["High Risk","Medium Risk","Low Risk"],
                values=[high, medium, low],
                hole=0.6,
                marker=dict(colors=["#E74C3C","#F39C12","#27AE60"],
                            line=dict(color="#0F1923", width=2)),
                textinfo="label+percent",
                textfont=dict(color="#E8E0D0", size=12)
            ))
            fig.add_annotation(text=f"<b>{total}</b><br>Students",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=18, color="#C9A84C"))
            fig.update_layout(**layout, height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            st.markdown("<div class='section-header'>Score Distribution</div>",
                        unsafe_allow_html=True)
            fig = go.Figure(go.Histogram(
                x=rdf["composite_score"], nbinsx=10,
                marker=dict(
                    color="#C9A84C",
                    line=dict(color="#0F1923", width=1)
                )
            ))
            fig.add_vline(x=70, line_dash="dash", line_color="#E74C3C",
                          annotation_text="High", annotation_font_color="#E74C3C")
            fig.add_vline(x=55, line_dash="dash", line_color="#F39C12",
                          annotation_text="Medium", annotation_font_color="#F39C12")
            fig.update_layout(**layout, height=300,
                              xaxis=dict(title="Composite Score",
                                         gridcolor="rgba(42,63,95,0.5)"),
                              yaxis=dict(title="Students",
                                         gridcolor="rgba(42,63,95,0.5)"))
            st.plotly_chart(fig, use_container_width=True)

        # Module comparison
        st.markdown("<div class='section-header'>Module Score Comparison</div>",
                    unsafe_allow_html=True)
        fig = go.Figure()
        for mod, color in [("module1_ai_score","#3498DB"),
                           ("module2_style_score","#C9A84C"),
                           ("module3_behavior_score","#27AE60")]:
            fig.add_trace(go.Box(
                y=rdf[mod],
                name=mod.replace("_score","").replace("_"," ").title(),
                marker_color=color, line_color=color
            ))
        fig.update_layout(**layout, height=300,
                          yaxis=dict(title="Score (%)",
                                     gridcolor="rgba(42,63,95,0.5)"))
        st.plotly_chart(fig, use_container_width=True)

        # Results table
        st.markdown("<div class='section-header'>Full Results Table</div>",
                    unsafe_allow_html=True)
        disp = rdf[[
            "student_id","student_name",
            "module1_ai_score","module2_style_score","module3_behavior_score",
            "behavior_label","composite_score","risk_level"
        ]].rename(columns={
            "student_id":"ID","student_name":"Name",
            "module1_ai_score":"M1 %","module2_style_score":"M2 %",
            "module3_behavior_score":"M3 %","behavior_label":"Behavior",
            "composite_score":"Score","risk_level":"Risk"
        })
        st.dataframe(disp, use_container_width=True, hide_index=True,
                     column_config={
                         "Score": st.column_config.ProgressColumn(
                             "Score", min_value=0, max_value=100, format="%.1f"
                         )
                     })

        import io
        buf = io.StringIO()
        disp.to_csv(buf, index=False)
        st.download_button("⬇️ Download Results", buf.getvalue(),
                           "results.csv", "text/csv")

        # High risk students
        high_df = rdf[rdf["risk_level"]=="High Risk"]
        if len(high_df) > 0:
            st.markdown("<div class='section-header'>🔴 High Risk Students</div>",
                        unsafe_allow_html=True)
            for _, row in high_df.iterrows():
                with st.expander(f"🔴 {row['student_name']} — Score: {row['composite_score']}"):
                    show_student_report(row.to_dict())

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif "📜" in page:
    st.markdown("<h2>Analysis History</h2>", unsafe_allow_html=True)
    st.markdown("<p class='info'>All previously analyzed students stored in database</p>",
                unsafe_allow_html=True)

    # add refresh button
    col_ref, col_empty = st.columns([1, 5])
    with col_ref:
        if st.button("🔄 Refresh"):
            st.rerun()

    try:
        all_records = get_all_results()

        records = all_records

        if not records:
            st.info("No records found. Analyze some students first.")
        else:
            # summary stats
            total  = len(records)
            high   = sum(1 for r in records if r["risk_level"] == "High Risk")
            medium = sum(1 for r in records if r["risk_level"] == "Medium Risk")
            low    = sum(1 for r in records if r["risk_level"] == "Low Risk")

            c1,c2,c3,c4 = st.columns(4)
            for col, val, lbl, color in zip(
                [c1,c2,c3,c4],
                [total, high, medium, low],
                ["Total Records","High Risk","Medium Risk","Low Risk"],
                ["#C9A84C","#FF6B6B","#FFB347","#6BCB77"]
            ):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value' style='color:{color};'>{val}</div>
                        <div class='metric-label'>{lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # search filter
            st.markdown("<div class='section-header'>Search and Filter</div>",
                        unsafe_allow_html=True)
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                search_name = st.text_input("Search by Name", placeholder="Student name...")
            with fc2:
                filter_risk = st.selectbox("Filter by Risk",
                                           ["All", "High Risk", "Medium Risk", "Low Risk"])
            with fc3:
                filter_behavior = st.selectbox("Filter by Behavior",
                                               ["All", "Anomaly", "Normal"])

            # apply filters
            filtered = records
            if search_name:
                filtered = [r for r in filtered
                            if search_name.lower() in r["student_name"].lower()]
            if filter_risk != "All":
                filtered = [r for r in filtered if r["risk_level"] == filter_risk]
            if filter_behavior != "All":
                filtered = [r for r in filtered if r["behavior_label"] == filter_behavior]

            st.markdown(f"<p class='info'>Showing {len(filtered)} of {total} records</p>",
                        unsafe_allow_html=True)

            # display records
            st.markdown("<div class='section-header'>Records</div>",
                        unsafe_allow_html=True)

            for record in filtered:
                rc     = risk_color(record["risk_level"])
                emoji  = {"High Risk":"🔴","Medium Risk":"🟡",
                          "Low Risk":"🟢"}.get(record["risk_level"],"⚪")

                # Convert UTC → IST
                try:
                    utc_time = datetime.fromisoformat(record["analyzed_at"].replace("Z", "+00:00"))
                    ist_time = utc_time + timedelta(hours=5, minutes=30)
                    display_time = ist_time.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    display_time = record["analyzed_at"]      

                with st.expander(
                    f"{emoji} {record['student_name']} ({record['student_id']}) "
                    f"— {record['risk_level']} — {display_time}"
                ):
                    dc1, dc2, dc3, dc4 = st.columns(4)
                    for col, val, lbl, clr in zip(
                        [dc1, dc2, dc3, dc4],
                        [record["composite_score"], record["module1_score"],
                         record["module2_score"], record["module3_score"]],
                        ["Composite","M1 Score","M2 Score","M3 Score"],
                        [rc, "#3498DB", "#C9A84C", "#27AE60"]
                    ):
                        with col:
                            st.markdown(f"""
                            <div class='metric-card' style='padding:0.8rem;'>
                                <div class='metric-value' style='color:{clr};
                                     font-size:1.5rem;'>{val}%</div>
                                <div class='metric-label'>{lbl}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    ic1, ic2 = st.columns(2)
                    with ic1:
                        st.markdown(f"""
                        <div style='background:#1B2A4A; border:1px solid #2A3F5F;
                                    border-radius:8px; padding:0.8rem; font-size:0.85rem;'>
                        <b style='color:#C9A84C;'>Grade Info</b><br>
                        <span style='color:#8A99B0;'>
                        G1: <b style='color:#E8E0D0;'>{record['G1']}</b> &nbsp;
                        G2: <b style='color:#E8E0D0;'>{record['G2']}</b> &nbsp;
                        G3: <b style='color:#E8E0D0;'>{record['G3']}</b><br>
                        Grade Jump: <b style='color:#E8E0D0;'>{record['grade_jump']}</b><br>
                        Behavior: <b style='color:{"#E74C3C" if record["behavior_label"]=="Anomaly" else "#27AE60"};'>
                        {record['behavior_label']}</b><br>
                        Reason: <span style='color:#C9A84C;'>{record.get("module3_reason", "N/A")}</span>
                        </span>
                        </div>
                        """, unsafe_allow_html=True)

                    with ic2:
                        copy_score = record.get("copy_score") or 0
                        matched_student = record.get("matched_student", "")

                        # Find matched student name
                        matched_name = "N/A"

                        if matched_student:
                            try:
                                all_records = get_all_results()

                                for r in all_records:
                                    if str(r["student_id"]) == str(matched_student):
                                        matched_name = r["student_name"]
                                        break

                            except:
                                matched_name = matched_student
                        
                        # Copy score color
                        if copy_score >= 80:
                            copy_color = "#E74C3C"

                        elif copy_score >= 60:
                            copy_color = "#F39C12"

                        else:
                            copy_color = "#27AE60"

                        st.markdown(f"""
                        <div style='background:#1B2A4A; border:1px solid #2A3F5F;
                                    border-radius:8px; padding:0.8rem; font-size:0.85rem;'>
                        <b style='color:#C9A84C;'>Student Info</b><br>
                        <span style='color:#8A99B0;'>
                        Writing Style: <b style='color:#E8E0D0;'>
                        {record.get("module2_label","N/A")}</b><br>
                        Absences: <b style='color:#E8E0D0;'>{record['absences']}</b><br>
                        Failures: <b style='color:#E8E0D0;'>{record['failures']}</b><br>
                        Possible Copy: <b style='color:{copy_color};'>{copy_score}%</b>&nbsp &nbsp &nbsp
                        Matched Student: <b style='color:#E8E0D0;'>{matched_name}</b>
                        </span>
                        </div>
                        """, unsafe_allow_html=True)

                    # show faculty note if exists
                    note_value = record.get("faculty_note")
                    if note_value and note_value.strip():
                        st.markdown(f"""
                        <div style='background:rgba(201,168,76,0.06); border:1px solid #2A3F5F; border-left:3px solid #C9A84C;
                            border-radius:8px; padding:0.8rem; margin-top:0.5rem; font-size:0.85rem;'>
                        <b style='color:#C9A84C;'>📝 Faculty Note:</b><br>
                        <span style='color:#E8E0D0;'>{record['faculty_note']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='background:#1B2A4A; border:1px solid #2A3F5F; border-radius:8px; padding:0.6rem 0.8rem;
                            margin-top:0.5rem; font-size:0.8rem; color:#8A99B0;'>
                        No faculty note added yet.
                        </div>
                        """, unsafe_allow_html=True)

                    # add/update note
                    new_note = st.text_area(
                        "Update Note", height=80,
                        value=record.get("faculty_note",""),
                        placeholder="Add faculty observation...",
                        key=f"hist_note_{record['id']}"
                    )
                    nc1, nc2 = st.columns([1,4])
                    with nc1:
                        if st.button("Save", key=f"hist_save_{record['id']}"):
                            update_note(record["id"], new_note)
                            st.success("Note updated!")
                            st.rerun()
                    with nc2:
                        if st.button("Delete Record",
                                     key=f"hist_del_{record['id']}",
                                     type="secondary"):
                            delete_record(record["id"])
                            st.warning("Record deleted.")
                            st.rerun()

            # download all history
            st.markdown("<br>", unsafe_allow_html=True)
            hist_df = pd.DataFrame(filtered)
            if not hist_df.empty:
                cols_to_show = ["student_id","student_name","composite_score",
                                "risk_level","behavior_label","module3_reason","module2_label",
                                "G1","G2","G3","grade_jump",
                                "absences","failures","faculty_note","analyzed_at"]
                hist_df = hist_df[[c for c in cols_to_show if c in hist_df.columns]]
                import io
                buf = io.StringIO()
                hist_df.to_csv(buf, index=False)
                st.download_button(
                    "⬇️ Download History CSV",
                    buf.getvalue(),
                    "analysis_history.csv",
                    "text/csv"
                )

    except Exception as e:
        st.error(f"Could not load history: {str(e)}")
        st.info("Make sure the database folder exists and you have analyzed at least one student.")