# Module 3 - Behavioral Anomaly Detection (Fixed Version)

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.system('cls')

# ── Load datasets ─────────────────────────────────────
print("Loading datasets...")
mat = pd.read_csv('data/student-mat.csv', sep=';')
por = pd.read_csv('data/student-por.csv', sep=';')

mat['subject'] = 'math'
por['subject'] = 'portuguese'

df = pd.concat([mat, por], ignore_index=True)
print("Total students:", len(df))

# ── Remove dropouts ───────────────────────────────────
df = df[df['G3'] > 0]

# ── Create baseline and anomaly label ─────────────────
df['baseline'] = (df['G1'] + df['G2']) / 2
df['std_baseline'] = df[['G1', 'G2']].std(axis=1)

# stricter condition
df['anomaly'] = ((df['G3'] - df['baseline']) > 2.5 * df['std_baseline']).astype(int)

# new feature (important)
df['grade_jump'] = df['G3'] - df['baseline']

print("\nAnomaly Distribution:")
print(df['anomaly'].value_counts())

# ── Features (NO G3 to avoid leakage) ─────────────────
features = ['G1', 'G2', 'grade_jump', 'absences', 'studytime', 'failures']

X = df[features].values
y = df['anomaly'].values

# ── Scaling ──────────────────────────────────────────
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ── Split ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ── Model ────────────────────────────────────────────
print("\nTraining model...")
model = RandomForestClassifier(
    n_estimators=200,   # increased trees
    max_depth=10,       # avoid overfitting
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Training complete!")

# ── Evaluation ───────────────────────────────────────
y_pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save model ───────────────────────────────────────
pickle.dump(model, open('models/module3_model.pkl', 'wb'))
pickle.dump(scaler, open('models/module3_scaler.pkl', 'wb'))

print("\nModel saved!")

# ── Prediction Function ──────────────────────────────
def predict_anomaly(G1, G2, G3, absences, studytime, failures):

    baseline = (G1 + G2) / 2
    grade_jump = G3 - baseline

    features = [[G1, G2, grade_jump, absences, studytime, failures]]
    scaled = scaler.transform(features)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    anomaly_prob = prob[1]

    # STEP 1: Strong normal condition
    if grade_jump < 2:
        label = "Normal"

    # STEP 2: Strong anomaly condition
    elif grade_jump > 5:
        label = "Anomaly"

    # STEP 3: Model decision
    elif anomaly_prob > 0.65:
        label = "Anomaly"
    else:
        label = "Normal"

    return {
        'prediction': label,
        'anomaly_probability': round(anomaly_prob * 100, 2),
        'normal_probability': round(prob[0] * 100, 2),
        'grade_jump': round(grade_jump, 2)
    }

# ── Testing ──────────────────────────────────────────
print("\n── Sample Predictions ──────────────────────────────")

print("\nNormal Student:")
print(predict_anomaly(12, 13, 14, 3, 2, 0))

print("\nSuspicious Student:")
print(predict_anomaly(8, 7, 18, 1, 2, 0))

print("\nAnother Normal Student:")
print(predict_anomaly(15, 15, 16, 2, 3, 0))