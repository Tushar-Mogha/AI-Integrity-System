# Module 3 - Behavioral Anomaly Detection
# We are trying to detect students whose grades suddenly jumped
# in an unusual way compared to their own previous performance
# Dataset - UCI Student Performance (Math + Portuguese)
# Team - Abhinandan, Stuti, Tushar

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.system('cls')

# 1. Load datasets
# -------------------------------

print("Loading datasets...")

mat = pd.read_csv('data/student-mat.csv', sep=';')
por = pd.read_csv('data/student-por.csv', sep=';')

print("Math students:", len(mat))
print("Portuguese students:", len(por))

# add subject column just to keep track
mat['subject'] = 'math'
por['subject'] = 'portuguese'

# combine both datasets
df = pd.concat([mat, por], ignore_index=True)

print("Total students after combining:", len(df))


# 2. Remove dropouts
# -------------------------------

# G3 = 0 usually means student dropped out
print("\nRemoving dropout students...")
df = df[df['G3'] > 0]
print("Students remaining:", len(df))


# 3. Create anomaly label
# -------------------------------

print("\nCreating anomaly labels...")

# baseline = average of G1 and G2
df['baseline'] = (df['G1'] + df['G2']) / 2

# standard deviation between G1 and G2
df['std_baseline'] = df[['G1', 'G2']].std(axis=1)

# anomaly condition:
# if G3 is much higher than expected based on previous performance
grade_jump = df['G3'] - df['baseline']

# additional behavioral features (help model learn patterns better)
df['trend'] = df['G2'] - df['G1']                 # improvement or decline
df['consistency'] = abs(df['G1'] - df['G2'])     # stability between exams
df['avg_score'] = (df['G1'] + df['G2']) / 2      # overall level of student

grade_jump = df['G3'] - df['baseline']

df['anomaly'] = (
    (abs(grade_jump) >= 3.5) |
    (df['absences'] > 10) |
    (df['failures'] > 1)
).astype(int)

# checking if both classes exist
if df['anomaly'].nunique() < 2:
    print("Error: Only one class present. Adjust labeling conditions.")
    exit()

print("\nAnomaly Distribution:")
print(df['anomaly'].value_counts())
print("Anomaly percentage:", round(df['anomaly'].mean() * 100, 2), "%")


# 4. Feature selection (IMPORTANT FIX)
# -------------------------------

# we DO NOT use grade_jump in training to avoid data leakage
features = ['G1', 'G2', 'trend', 'consistency', 'avg_score', 'absences', 'studytime', 'failures']

X = df[features].values
y = df['anomaly'].values


# 5. Scaling
# -------------------------------

print("\nScaling features...")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# 6. Train-test split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# 7. Train model
# -------------------------------

print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training complete!")


# 8. Evaluation
# -------------------------------

y_pred = model.predict(X_test)

print("\n── Results ────────────────────────────────────────")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 9. Save model
# -------------------------------

print("\nSaving model...")

pickle.dump(model, open('models/module3_model.pkl', 'wb'))
pickle.dump(scaler, open('models/module3_scaler.pkl', 'wb'))

print("Saved!")
print("\nModule 3 done!")


# 10. Prediction function
# -------------------------------

def predict_anomaly(G1, G2, G3, absences, studytime, failures):

    # -----------------------------
    # Step 1: Compute behavioral features
    # -----------------------------
    baseline = (G1 + G2) / 2
    grade_jump = G3 - baseline
    jump_abs = abs(grade_jump)

    trend = G2 - G1
    consistency = abs(G1 - G2)
    avg_score = baseline

    # -----------------------------
    # Step 2: Model prediction (support only)
    # -----------------------------
    features_input = [[G1, G2, trend, consistency, avg_score, absences, studytime, failures]]
    scaled = scaler.transform(features_input)

    prob = model.predict_proba(scaled)[0]
    model_anomaly_prob = prob[1]

    # -----------------------------
    # Step 3: RULE-BASED CORE LOGIC (PRIMARY)
    # -----------------------------

    # Strong anomaly (extreme jump/dip)
    if grade_jump >= 6 or grade_jump <= -4:
        label = "Anomaly"
        final_prob = max(model_anomaly_prob, 0.90)
    
    elif consistency >= 5 and jump_abs >= 4:
        label = "Anomaly"
        final_prob = max(model_anomaly_prob, 0.75)

    # Moderate anomaly (beyond threshold)
    elif jump_abs >= 3.5 and consistency <= 1:
        label = "Anomaly"
        final_prob = max(model_anomaly_prob, 0.70)

    # Behavior-based anomaly
    elif absences > 10 or failures > 1:
        label = "Anomaly"
        final_prob = max(model_anomaly_prob, 0.75)

    # # Consistency anomaly (sudden change after stable past)
    # elif consistency <= 2 and 3 < jump_abs < 3.5:
    #     label = "Anomaly"
    #     final_prob = max(model_anomaly_prob, 0.60)

    # Normal gradual change
    else:
        label = "Normal"
        final_prob = min(model_anomaly_prob, 0.30)
    
    # -----------------------------
    # Step 3.5: Explanation (reason)
    # -----------------------------

    if label == "Anomaly":

        if grade_jump >= 6:
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

    # -----------------------------
    # Step 4: Final probability cleanup
    # -----------------------------
    final_prob = round(final_prob * 100, 2)

    return {
        'prediction': label,
        'anomaly_probability': float(final_prob),
        'normal_probability': float(round(100 - final_prob, 2)),
        'grade_jump': round(grade_jump, 2),
        'reason': reason
    }

print("\n── Structured Behavior Testing (Module 3) ─────────────────────────────")

print(f"{'Scenario':<45} {'G1':>4} {'G2':>4} {'G3':>4} {'Jump':>6} {'Anom%':>8} {'Label':<10} {'Reason'}")
print("-" * 115)

test_cases = [

    # ─────────── CONSISTENT PERFORMANCE ───────────
    ("Consistent average",              12, 12, 12, 3, 2, 0),
    ("Consistent high",                 18, 18, 18, 1, 3, 0),
    ("Consistent low",                   5,  5,  5, 4, 1, 1),

    # ─────────── GRADUAL INCREASE ───────────
    ("Gradual increase small",          10, 11, 12, 2, 2, 0),
    ("Gradual increase moderate",        8, 10, 13, 3, 2, 0),
    ("Gradual increase near threshold", 10, 11, 13.4, 2, 2, 0),

    # ─────────── GRADUAL DECREASE ───────────
    ("Gradual decrease small",          14, 13, 12, 2, 2, 0),
    ("Gradual decrease moderate",       16, 14, 12, 3, 2, 0),
    ("Gradual decrease near threshold", 15, 14, 11.6, 2, 2, 0),

    # ─────────── SUDDEN INCREASE ───────────
    ("Sudden jump (clear anomaly)",      8,  7, 18, 1, 2, 0),
    ("Weak → topper",                   5,  5, 19, 0, 3, 0),
    ("Average → very high",            10, 10, 18, 2, 2, 0),

    # ─────────── SUDDEN DECREASE ───────────
    ("Sudden drop (clear anomaly)",     18, 17,  6, 2, 2, 0),
    ("Topper → low",                    20, 20,  8, 3, 2, 0),
    ("Moderate → very low",             15, 17,  10, 3, 2, 0),

    # ─────────── FLUCTUATIONS ───────────
    ("Zigzag (high-low-high)",          18,  6, 17, 4, 2, 0),
    ("Zigzag (low-high-low)",            5, 17,  6, 3, 2, 0),
    ("Inconsistent pattern",            14,  9, 15, 5, 2, 1),

    # ─────────── THRESHOLD EDGE CASES ───────────
    ("Exactly +3.5 threshold",          10, 10, 13.5, 2, 2, 0),
    ("Exactly -3.5 threshold",          15, 15, 11.5, 2, 2, 0),
    ("Just below +3.5",                 10, 10, 13.4, 2, 2, 0),
    ("Just below -3.5",                 15, 15, 11.6, 2, 2, 0),

    # ─────────── MIXED REALISTIC CASES ───────────
    ("Gradual → sudden jump",           10, 12, 18, 2, 2, 0),
    ("Stable → sudden drop",            14, 14,  7, 3, 2, 0),
    ("Increase → plateau",              10, 12, 12, 2, 2, 0),
    ("Decrease → recovery",             15, 12, 14, 2, 2, 0),
]

for desc, G1, G2, G3, absences, studytime, failures in test_cases:
    result = predict_anomaly(G1, G2, G3, absences, studytime, failures)

    print(f"{desc:<45} {G1:>4} {G2:>4} {G3:>4} "
          f"{result['grade_jump']:>6.1f} "
          f"{result['anomaly_probability']:>7.1f}% "
          f"{result['prediction']:<10} "
          f"{result['reason']}")