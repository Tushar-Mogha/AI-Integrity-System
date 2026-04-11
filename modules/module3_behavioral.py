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

# loading both datasets
print("Loading datasets...")
mat = pd.read_csv('data/student-mat.csv', sep=';')
por = pd.read_csv('data/student-por.csv', sep=';')
print("Math students:", len(mat))
print("Portuguese students:", len(por))

# adding subject column to know which dataset each student came from
mat['subject'] = 'math'
por['subject'] = 'portuguese'

# combining both datasets into one
df = pd.concat([mat, por], ignore_index=True)
print("Total students after combining:", len(df))

# removing students who dropped out
# G3 = 0 means the student left the course, not that they cheated
print("\nRemoving dropout students...")
df = df[df['G3'] > 0]
print("Students remaining:", len(df))

# creating the anomaly label
# we calculate each student's baseline from G1 and G2
# if G3 jumps more than 2.5 standard deviations above baseline
# we flag that student as suspicious
print("\nCreating anomaly labels...")
df['baseline']     = (df['G1'] + df['G2']) / 2
df['std_baseline'] = df[['G1', 'G2']].std(axis=1)
df['anomaly']      = ((df['G3'] - df['baseline']) > 2.5 * df['std_baseline']).astype(int)

# grade_jump tells us exactly how much the grade jumped
df['grade_jump'] = df['G3'] - df['baseline']

# how consistent were G1 and G2
df['grade_consistency'] = abs(df['G1'] - df['G2'])

# grade jump relative to student's performance level
# a jump of 3 for a weak student is different from a jump of 3 for a strong student
df['relative_jump'] = df['grade_jump'] / (df['baseline'] + 1)

print("\nAnomaly Distribution:")
print(df['anomaly'].value_counts())
print("Anomaly percentage:", round(df['anomaly'].mean() * 100, 2), "%")

# selecting features for training
# we use grade_jump and grade_consistency along with academic features
# grade_jump captures the suspicious behaviour we are looking for
# grade_consistency tells us how stable the student was before the jump
features = ['G1', 'G2', 'grade_jump', 'grade_consistency', 'absences', 'studytime', 'failures']
X = df[features].values
y = df['anomaly'].values

# scaling features between 0 and 1
print("\nScaling features...")
scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# splitting data - 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# training Random Forest model
# class_weight balanced handles the imbalance between normal and anomaly students
# max_depth 10 prevents the model from overfitting on small data
print("\nTraining model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Training complete!")

# checking model performance
y_pred = model.predict(X_test)

print("\n── Results ────────────────────────────────────────")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# saving model files
print("\nSaving model...")
pickle.dump(model,  open('models/module3_model.pkl', 'wb'))
pickle.dump(scaler, open('models/module3_scaler.pkl', 'wb'))
print("Saved!")
print("\nModule 3 done!")

# prediction function for new students
# we use a combination of model prediction and manual rules
# to make sure obvious cases are handled correctly
def predict_anomaly(G1, G2, G3, absences, studytime, failures):
    baseline          = (G1 + G2) / 2
    grade_jump        = G3 - baseline
    grade_consistency = abs(G1 - G2)

    features_input = [[G1, G2, grade_jump, grade_consistency,
                       absences, studytime, failures]]
    scaled         = scaler.transform(features_input)
    pred           = model.predict(scaled)[0]
    prob           = model.predict_proba(scaled)[0]
    anomaly_prob   = prob[1]

    if grade_jump < 2:
        label = "Normal"
    elif grade_jump >= 7:
        label = "Anomaly"
    elif 3 <= grade_jump < 7:
        if failures > 0 or absences > 8 or anomaly_prob > 0.25:
            label = "Anomaly"
        else:
            label = "Normal"
    else:
        label = "Normal"

    return {
        'prediction'         : label,
        'anomaly_probability': round(anomaly_prob * 100, 2),
        'normal_probability' : round(prob[0] * 100, 2),
        'grade_jump'         : round(grade_jump, 2)
    }

# testing with sample students
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