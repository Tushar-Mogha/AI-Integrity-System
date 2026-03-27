# Writing Style Analysis - Module 2
# This module extracts writing style features from essays
# and classifies them as AI-generated or Human-written
# Dataset used: DAIGT-V2 (Kaggle)

import os
import string
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.system('cls')

# ── Loading the dataset ───────────────────────────────────────────────────────
print("Loading DAIGT-V2 dataset...")
data = pd.read_csv('data/train_v2_drcat_02(Model_1).csv')
print(f"Total essays loaded: {len(data)}")
print(f"Human essays: {len(data[data['label']==0])}")
print(f"AI essays: {len(data[data['label']==1])}")


# ── Feature extraction ────────────────────────────────────────────────────────
# We extract 7 writing style features from each essay
# These features capture how a person writes - not what they write about

def get_writing_features(essay):
    
    # return zeros if essay is empty or not a string
    if not isinstance(essay, str) or len(essay.strip()) == 0:
        return [0, 0, 0, 0, 0, 0, 0]
    
    essay_clean = essay.strip()
    
    # clean words - remove punctuation and lowercase
    raw_words = essay_clean.split()
    words = [w.strip(string.punctuation).lower() for w in raw_words]
    words = [w for w in words if len(w) > 0]
    
    # split into sentences - treat ! ? ; : same as full stop
    for char in ['!', '?', ';', ':']:
        essay_clean = essay_clean.replace(char, '.')
    sentences = [s.strip() for s in essay_clean.split('.') if len(s.strip()) > 5]
    
    # split into paragraphs
    paragraphs = [p.strip() for p in essay.split('\n') if len(p.strip()) > 10]
    
    # check for transition/linking words
    linking_words = [
        'however', 'therefore', 'moreover', 'furthermore',
        'although', 'nevertheless', 'consequently', 'additionally',
        'meanwhile', 'otherwise', 'similarly', 'thus'
    ]
    essay_words = essay.lower().split()
    linking_count = sum(1 for w in linking_words if w in essay_words)
    
    # count meaningful punctuation only
    meaningful_punct = set('.,!?;:')
    punct_count = sum(1 for ch in essay if ch in meaningful_punct)
    
    # feature 1 - how long are the words on average
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    
    # feature 2 - how long are the sentences on average
    avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    
    # feature 3 - vocabulary richness (more unique words = richer vocabulary)
    vocab_richness = len(set(words)) / len(words) if words else 0
    
    # feature 4 - punctuation usage
    # feature 5 - number of paragraphs
    para_count = len(paragraphs)
    
    # feature 6 - use of linking/transition words
    # feature 7 - how often capital letters are used
    capital_ratio = sum(1 for c in essay if c.isupper()) / len(essay) if essay else 0

    return [
        avg_word_len,
        avg_sent_len,
        vocab_richness,
        punct_count,
        para_count,
        linking_count,
        capital_ratio
    ]


# apply feature extraction to all essays
print("\nExtracting writing style features from all essays...")
print("This will take about a minute...")
extracted = data['text'].apply(get_writing_features)

style_df = pd.DataFrame(extracted.tolist(), columns=[
    'avg_word_len',
    'avg_sent_len',
    'vocab_richness',
    'punct_count',
    'para_count',
    'linking_count',
    'capital_ratio'
])
style_df['label'] = data['label']

print(f"Features extracted successfully!")
print(f"\nSample features:")
print(style_df.head(3))


# ── Removing bad data ─────────────────────────────────────────────────────────
# removing essays with unrealistic word/sentence lengths
# these are likely corrupted or non-essay entries
print("\nCleaning outliers...")
style_df = style_df[style_df['avg_word_len'] < 20]
style_df = style_df[style_df['avg_sent_len'] < 200]
clean_data = data.loc[style_df.index]
print(f"Essays after cleaning: {len(style_df)}")


# ── TF-IDF features ───────────────────────────────────────────────────────────
# TF-IDF captures word patterns in the essays
# AI writing tends to use certain words more than humans
# we use 500 most important word patterns (unigrams and bigrams)
print("\nApplying TF-IDF on essay text...")
tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(clean_data['text'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")


# ── Combining features ────────────────────────────────────────────────────────
# scale the style features first
scaler = StandardScaler()
style_scaled = scaler.fit_transform(style_df.drop('label', axis=1))

# combine tfidf + style features into one feature matrix
X = hstack([tfidf_matrix, sp.csr_matrix(style_scaled)])
y = style_df['label'].values
print(f"\nFinal feature matrix shape: {X.shape}")


# ── Train test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# ── Model training ────────────────────────────────────────────────────────────
# using Random Forest - works well with mixed feature types
# n_jobs=-1 uses all CPU cores to speed up training
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Training complete!")


# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print("\n── Model Performance ──────────────────────────────")
print(f"Accuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ── Saving the model ──────────────────────────────────────────────────────────
# saving model, scaler and tfidf so we can use them later in the dashboard
# without having to retrain everything from scratch
print("\nSaving model files...")
pickle.dump(model, open('models/module2_model.pkl', 'wb'))
pickle.dump(scaler, open('models/module2_scaler.pkl', 'wb'))
pickle.dump(tfidf, open('models/module2_tfidf.pkl', 'wb'))
print("Model saved to models/ folder")
print("\nModule 2 complete!")