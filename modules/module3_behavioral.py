# Behavioral Anomaly Detection - Module 3
# Detects unusual grade patterns in student performance
# Dataset: UCI Student Performance Dataset

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.system('cls')

# ── Load datasets ─────────────────────────────────────────────────────────────
print("Loading UCI Student Performance datasets...")
mat = pd.read_csv('data/student-mat.csv', sep=';')
por = pd.read_csv('data/student-por.csv', sep=';')
print(f"Math students: {len(mat)}")
print(f"Portuguese students: {len(por)}")

# ── Combine datasets ──────────────────────────────────────────────────────────
print("\nCombining both datasets...")
mat['subject'] = 'math'
por['subject'] = 'portuguese'
df = pd.concat([mat, por], ignore_index=True)
print(f"Total students: {len(df)}")

# ── Remove dropouts ───────────────────────────────────────────────────────────
print("\nRemoving dropout students (G3=0)...")
df = df[df['G3'] > 0]
print(f"Students after removing dropouts: {len(df)}")

# ── Engineer anomaly label ────────────────────────────────────────────────────
# flag students where G3 jumps significantly above G1 and G2 baseline
print("\nEngineering anomaly labels...")
df['baseline'] = (df['G1'] + df['G2']) / 2
df['std_baseline'] = df[['G1', 'G2']].std(axis=1)
df['anomaly'] = ((df['G3'] - df['baseline']) > 1.5 * df['std_baseline']).astype(int)

print(f"\nAnomaly Distribution:")
print(df['anomaly'].value_counts())
print(f"Anomaly percentage: {round(df['anomaly'].mean()*100, 2)}%")

# ── Prepare features ──────────────────────────────────────────────────────────
features = ['G1', 'G2', 'G3', 'absences', 'studytime', 'failures']
X = df[features].values
y = df['anomaly'].values

# scale features
print("\nScaling features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ── Train Random Forest ───────────────────────────────────────────────────────
# using class_weight balanced to handle class imbalance
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Training complete!")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print(f"\nAccuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save model ────────────────────────────────────────────────────────────────
print("\nSaving model...")
pickle.dump(model, open('models/module3_model.pkl', 'wb'))
pickle.dump(scaler, open('models/module3_scaler.pkl', 'wb'))
print("Model saved to models/ folder")
print("\nModule 3 complete!")

# testing with sample students to verify predictions
model_loaded  = pickle.load(open('models/module3_model.pkl', 'rb'))
scaler_loaded = pickle.load(open('models/module3_scaler.pkl', 'rb'))

def predict_anomaly(G1, G2, G3, absences, studytime, failures):
    features = [[G1, G2, G3, absences, studytime, failures]]
    scaled   = scaler_loaded.transform(features)
    pred     = model_loaded.predict(scaled)[0]
    prob     = model_loaded.predict_proba(scaled)[0]
    label    = "Anomaly" if pred == 1 else "Normal"
    return {
        'prediction'         : label,
        'anomaly_probability': round(prob[1] * 100, 2),
        'normal_probability' : round(prob[0] * 100, 2)
    }

print("\n── Sample Predictions ──────────────────────────────")

# normal student - consistent grades
print("\nNormal Student (G1=12, G2=13, G3=14):")
print(predict_anomaly(12, 13, 14, 3, 2, 0))

# suspicious student - sudden grade spike
print("\nSuspicious Student (G1=8, G2=7, G3=18):")
print(predict_anomaly(8, 7, 18, 1, 2, 0))

# another normal student
print("\nNormal Student (G1=15, G2=15, G3=16):")
print(predict_anomaly(15, 15, 16, 2, 3, 0))