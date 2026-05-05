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
    (grade_jump >= 2) |
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
    # calculate baseline and behavior indicators
    baseline = (G1 + G2) / 2
    grade_jump = G3 - baseline
    grade_consistency = abs(G1 - G2)
    trend = G2 - G1
    consistency = abs(G1 - G2)
    avg_score = (G1 + G2) / 2

    # model only sees historical features
    features_input = [[G1, G2, trend, consistency, avg_score,absences, studytime, failures]]
    scaled = scaler.transform(features_input)

    prob = model.predict_proba(scaled)[0]
    anomaly_prob = prob[1]

    # step 1: handle extreme cases FIRST
    if grade_jump >= 6:
        label = "Anomaly"

    elif grade_jump >= 4:
        label = "Anomaly"
    
    elif grade_jump <= -5:
        label = "Anomaly"

    # step 2: model decision
    elif anomaly_prob >= 0.40:
        label = "Anomaly"

    elif anomaly_prob <= 0.25:
        label = "Normal"

    # step 3: rules
    else:
        if grade_jump >= 3:
            label = "Anomaly"

        elif grade_jump <= -4 and (failures > 0 or absences > 6):
            label = "Anomaly"

        elif grade_consistency >= 6 and grade_jump > 2:
            label = "Anomaly"

        elif failures > 0 or absences > 8:
            label = "Anomaly"

        else:
            label = "Normal"

    return {
        'prediction': label,
        'anomaly_probability': round(anomaly_prob * 100, 2),
        'normal_probability': round(prob[0] * 100, 2),
        'grade_jump': round(grade_jump, 2)
    }

#testing with sample students
print("\n── Sample Predictions ──────────────────────────────")

print("\nNormal Student (G1=12, G2=13, G3=14):")
print(predict_anomaly(12, 13, 14, 3, 2, 0))

print("\nSuspicious Student (G1=8, G2=7, G3=18):")
print(predict_anomaly(8, 7, 18, 1, 2, 0))

print("\nAnother Normal Student (G1=15, G2=15, G3=16):")
print(predict_anomaly(15, 15, 16, 2, 3, 0))

print("\nNormal Student (G1=10, G2=11, G3=12):")
print(predict_anomaly(10, 11, 12, 2, 2, 0))

print("\nConsistent Student (G1=15, G2=15, G3=15):")
print(predict_anomaly(15, 15, 15, 1, 3, 0))

print("\nSlight Increase (G1=12, G2=13, G3=15):")
print(predict_anomaly(12, 13, 15, 3, 2, 0))

print("\nSlight Drop (G1=14, G2=13, G3=12):")
print(predict_anomaly(14, 13, 12, 2, 2, 0))

print("\nSudden Jump (G1=8, G2=7, G3=18):")
print(predict_anomaly(8, 7, 18, 1, 2, 0))

print("\nWeak to Topper (G1=6, G2=5, G3=17):")
print(predict_anomaly(6, 5, 17, 0, 3, 0))

print("\nHigh Absence but High Marks (G1=10, G2=10, G3=18):")
print(predict_anomaly(10, 10, 18, 15, 1, 0))

print("\nFailures but High G3 (G1=7, G2=6, G3=16):")
print(predict_anomaly(7, 6, 16, 3, 2, 2))

print("\nBorderline Jump (G1=10, G2=10, G3=14):")
print(predict_anomaly(10, 10, 14, 2, 2, 0))

print("\nModerate Jump (G1=11, G2=10, G3=15):")
print(predict_anomaly(11, 10, 15, 3, 2, 0))

# ── Thorough Testing ──────────────────────────────────────────────────────────
# print("\n── Thorough Module 3 Testing ───────────────────────────────────────")
# print(f"{'Scenario':<35} {'G1':>4} {'G2':>4} {'G3':>4} {'Jump':>6} {'Anomaly%':>9} {'Label':<10}")
# print("-" * 80)

# scenarios = [
#     # (description, G1, G2, G3, absences, studytime, failures)

#     # Consistent performance
#     ("Consistent average",          10, 10, 10, 3, 2, 0),
#     ("Consistent high performer",   17, 18, 18, 1, 4, 0),
#     ("Consistent low performer",     5,  6,  5, 8, 1, 1),

#     # Gradual improvement
#     ("Gradual improvement",         10, 12, 14, 2, 3, 0),
#     ("Slow steady climb",            8,  9, 11, 3, 2, 0),

#     # Gradual decline
#     ("Gradual decline",             15, 13, 11, 4, 2, 0),
#     ("Sharp decline",               16, 14,  8, 6, 1, 0),

#     # Sudden suspicious jumps
#     ("Weak to topper sudden",        5,  4, 18, 1, 2, 0),
#     ("Below avg to excellent",       8,  7, 17, 0, 1, 0),
#     ("Average to perfect",          10, 10, 20, 2, 2, 0),

#     # High fluctuations
#     ("High then low then high",     18,  6, 19, 5, 2, 0),
#     ("Low then high then low",       5, 17,  4, 3, 2, 0),
#     ("Zigzag pattern",              13, 12 , 6, 5, 2, 4),

#     # Borderline cases
#     ("Borderline jump",             10, 10, 14, 2, 2, 0),
#     ("Slight improvement",          12, 13, 15, 2, 3, 0),
#     ("Very slight improvement",     14, 14, 15, 1, 3, 0),

#     # With absences and failures
#     ("Jump + high absences",         8,  7, 17, 15, 1, 0),
#     ("Jump + failures",              7,  6, 16,  3, 2, 2),
#     ("Jump + absences + failures",   6,  5, 18, 12, 1, 1),

#     # Perfect scores
#     ("Already perfect stays",       20, 20, 20,  0, 4, 0),
#     ("Perfect then drops",          20, 20,  8,  8, 1, 0),
# ]

# for desc, G1, G2, G3, absences, studytime, failures in scenarios:
#     result     = predict_anomaly(G1, G2, G3, absences, studytime, failures)
#     grade_jump = result['grade_jump']
#     anomaly_p  = result['anomaly_probability']
#     label      = result['prediction']
#     print(f"{desc:<35} {G1:>4} {G2:>4} {G3:>4} {grade_jump:>6.1f} {anomaly_p:>8.1f}% {label:<10}")