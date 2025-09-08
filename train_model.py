# train_model.py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

CSV_PATH = "Student_performance_data.csv"
MODEL_PATH = "student_performance_model.pkl"

#  1) Load data ---
df = pd.read_csv(CSV_PATH)

# 2) Normalise / map categories to match app.py encodings ---

# gender: Female -> 1 else 0
def enc_gender(x: str) -> int:
    return 1 if str(x).strip().lower() == "female" else 0

# ethnicity: Caucasian -> 1, Asian/Aisan -> 2, everything else -> 0
def enc_ethnicity(x: str) -> int:
    s = str(x).strip().lower()
    if s == "caucasian":
        return 1
    if s in ("asian", "aisan"):   # dataset has a misspelling "Aisan"
        return 2
    return 0

# parental education: High School -> 0, Some College -> 1, Bachelor's -> 2
# map anything else to nearest bucket (keeps more rows usable)
def enc_parent_edu(x: str) -> int:
    s = str(x).strip().lower()
    if s == "high school":
        return 0
    if s == "some college":
        return 1
    if s == "bachelor's":
        return 2
    # reasonable fallback mapping
    if s in ("master's", "masters"):
        return 2
    if s in ("no education", ""):
        return 0
    return 0

# parental support: Low -> 0, Moderate -> 1, High -> 2 (fold extras)
def enc_parent_support(x: str) -> int:
    s = str(x).strip().lower()
    if s == "low":
        return 0
    if s == "moderate":
        return 1
    if s == "high":
        return 2
    if s in ("very high",):
        return 2
    if s in ("no support",):
        return 0
    return 1

# yes/no to 1/0
def yes_no(x: str) -> int:
    return 1 if str(x).strip().lower() == "yes" else 0

# 3) Build features in the EXACT order app.py expects ---
# Order required by your app:
# [age, gender(F=1), ethnicity, parentalEducation, studyTime, absences,
#  tutoring, parentalSupport, extracurricular, sports, music, volunteering, gpa]
def row_to_features(row):
    return [
        float(row["Age"]),
        enc_gender(row["Gender"]),
        enc_ethnicity(row["Ethnicity"]),
        enc_parent_edu(row["ParentalEducation"]),
        float(row["StudyTimeWeekly"]),
        int(row["Absences"]),
        yes_no(row["Tutoring"]),
        enc_parent_support(row["ParentalSupport"]),
        yes_no(row["Extracurricular"]),
        yes_no(row["Sports"]),
        yes_no(row["Music"]),
        yes_no(row["Volunteering"]),
        float(row["GPA"]),
    ]

required_cols = [
    "Age","Gender","Ethnicity","ParentalEducation","StudyTimeWeekly","Absences",
    "Tutoring","ParentalSupport","Extracurricular","Sports","Music","Volunteering","GPA","GradeClass"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV is missing columns: {missing}")

# Drop rows with any essential numeric NA to avoid crashes
df = df.dropna(subset=["Age","StudyTimeWeekly","Absences","GPA"])

X = np.vstack(df.apply(row_to_features, axis=1).values)
y = df["GradeClass"].astype(str).values  # labels are letters like A/B/C/D/F

# --- 4) Train/test split just to sanity-check ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Simple, readable model (same family your app expects)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# --- 5) Quick metrics so you know it's fine ---
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Validation accuracy: {acc:.3f}")
print(classification_report(y_test, pred))

# --- 6) Save model where app.py expects it ---
joblib.dump(clf, MODEL_PATH)
print(f"Saved model to {os.path.abspath(MODEL_PATH)}")
