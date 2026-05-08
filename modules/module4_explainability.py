# Module 4 - SHAP Explainability
# Explains why a student was flagged using SHAP values
# Applied on Module 2 (Random Forest) as it is the most interpretable
# Team - Abhinandan, Stuti, Tushar

import shap
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import hstack
import scipy.sparse as sp
import string
import os

# ── Load Module 2 Models ──────────────────────────────────────────────────────
print("Loading Module 2 models for SHAP analysis...")
m2_model  = pickle.load(open("models/module2_model.pkl",  "rb"))
m2_scaler = pickle.load(open("models/module2_scaler.pkl", "rb"))
m2_tfidf  = pickle.load(open("models/module2_tfidf.pkl",  "rb"))
print("Models loaded!")

os.makedirs("outputs", exist_ok=True)

# ── Writing Features ──────────────────────────────────────────────────────────
STYLE_FEATURES = [
    'avg_word_len', 'avg_sent_len', 'vocab_richness',
    'punct_count', 'para_count', 'linking_count', 'capital_ratio'
]

def get_writing_features(essay):
    if not isinstance(essay, str) or len(essay.strip()) == 0:
        return [0] * 7
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

# ── Prepare Features for SHAP ─────────────────────────────────────────────────
def prepare_features(essay):
    features     = get_writing_features(essay)
    style_input  = pd.DataFrame([features], columns=STYLE_FEATURES)
    style_scaled = m2_scaler.transform(style_input)
    tfidf_input  = m2_tfidf.transform([essay])
    X_input      = hstack([tfidf_input, sp.csr_matrix(style_scaled)])
    return X_input, features

# ── Get Feature Names ─────────────────────────────────────────────────────────
def get_feature_names():
    tfidf_names = m2_tfidf.get_feature_names_out().tolist()
    return tfidf_names + STYLE_FEATURES

# ── SHAP Analysis on Style Features Only ─────────────────────────────────────
# We apply SHAP on style features only as TF-IDF has 500 features
# Style features are more interpretable for faculty

def get_shap_explanation(essay, student_name="Student"):
    features = get_writing_features(essay)
    style_df = pd.DataFrame([features], columns=STYLE_FEATURES)
    style_scaled = m2_scaler.transform(style_df)

    # use a simple explainer on just the style features
    # train background data using style features
    background = np.zeros((1, len(STYLE_FEATURES)))
    explainer  = shap.Explainer(
        lambda x: m2_model.predict_proba(
            hstack([m2_tfidf.transform([essay]), sp.csr_matrix(x)])
        ),
        background,
        feature_names=STYLE_FEATURES
    )

    shap_values = explainer(style_scaled)
    return shap_values, features

# ── Generate SHAP Bar Plot ────────────────────────────────────────────────────
def generate_shap_plot(essay, student_name="Student", save_path=None):
    features     = get_writing_features(essay)
    style_df     = pd.DataFrame([features], columns=STYLE_FEATURES)
    style_scaled = m2_scaler.transform(style_df)

    # calculate feature importance manually for interpretability
    # use permutation based approach for style features
    base_prob  = m2_model.predict_proba(
        hstack([m2_tfidf.transform([essay]),
                sp.csr_matrix(style_scaled)])
    )[0][1]

    importances = []
    for i, feat in enumerate(STYLE_FEATURES):
        modified        = style_scaled.copy()
        modified[0][i]  = 0
        modified_prob   = m2_model.predict_proba(
            hstack([m2_tfidf.transform([essay]),
                    sp.csr_matrix(modified)])
        )[0][1]
        importances.append(base_prob - modified_prob)

    # creating plot
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0F1923')
    ax.set_facecolor('#1B2A4A')

    colors = ['#E74C3C' if v > 0 else '#27AE60' for v in importances]
    bars   = ax.barh(STYLE_FEATURES, importances, color=colors, edgecolor='#2A3F5F')

    ax.set_xlabel('Impact on AI Probability', color='#E8E0D0', fontsize=11)
    ax.set_title(f'Writing Style Feature Impact — {student_name}',
                 color='#C9A84C', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(colors='#8A99B0')
    ax.spines['bottom'].set_color('#2A3F5F')
    ax.spines['left'].set_color('#2A3F5F')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=0, color='#8A99B0', linewidth=0.8, linestyle='--')

    for bar, val in zip(bars, importances):
        ax.text(val + 0.001 if val >= 0 else val - 0.001,
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}',
                va='center',
                ha='left' if val >= 0 else 'right',
                color='#E8E0D0', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0F1923')
        plt.close()
        return save_path
    else:
        return fig

# ── Simple Text Explanation ───────────────────────────────────────────────────
def get_text_explanation(essay, ai_score):
    features = get_writing_features(essay)
    feat_df  = pd.DataFrame([features], columns=STYLE_FEATURES)

    explanations = []

    if feat_df['linking_count'].values[0] > 3:
        explanations.append(
            f"High use of linking words ({int(feat_df['linking_count'].values[0])}) "
            f"— AI text commonly uses transition words like 'furthermore', 'moreover'"
        )
    if feat_df['avg_word_len'].values[0] > 6:
        explanations.append(
            f"Long average word length ({feat_df['avg_word_len'].values[0]:.1f} chars) "
            f"— AI tends to use more complex vocabulary"
        )
    if feat_df['avg_sent_len'].values[0] > 25:
        explanations.append(
            f"Long average sentence length ({feat_df['avg_sent_len'].values[0]:.1f} words) "
            f"— AI tends to write longer, more structured sentences"
        )
    if feat_df['vocab_richness'].values[0] > 0.7:
        explanations.append(
            f"High vocabulary richness ({feat_df['vocab_richness'].values[0]:.2f}) "
            f"— AI uses diverse vocabulary more consistently than humans"
        )
    if not explanations:
        explanations.append(
            f"TF-IDF word pattern analysis flagged this essay with "
            f"{ai_score}% AI probability based on word usage patterns"
        )

    return explanations


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    human_essay = "I think phones are really dangerous when people use them while driving. My uncle got into an accident because he was texting."
    ai_essay    = "The proliferation of mobile telecommunication devices has engendered substantial deliberation regarding their utilization in contemporary academic environments. Furthermore, empirical evidence consistently demonstrates cognitive impairment."

    print("\n── SHAP Analysis ────────────────────────────────────")
    print("\nHuman Essay Explanations:")
    for e in get_text_explanation(human_essay, 29):
        print(f"  - {e}")

    print("\nAI Essay Explanations:")
    for e in get_text_explanation(ai_essay, 92):
        print(f"  - {e}")

    print("\nGenerating SHAP plot for AI essay...")
    generate_shap_plot(ai_essay, "Test Student", "outputs/shap_test.png")
    print("Plot saved to outputs/shap_test.png")
    print("\nModule 4 complete!")