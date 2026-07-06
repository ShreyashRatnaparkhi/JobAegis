# ==========================================================
# JobAegis - Fake Job Posting Detection
# Author: Shreyash
# ==========================================================

# ===================== IMPORT LIBRARIES ====================

import re
import os
import joblib
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("        JOBAEGIS - FAKE JOB DETECTION")
print("=" * 60)

print("\nLoading Dataset...")

df = pd.read_csv("dataset/fake_job_postings.csv")

print("\nDataset Loaded Successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Variable Distribution:")
print(df["fraudulent"].value_counts())

print("\nPercentage Distribution:")
print(round(df["fraudulent"].value_counts(normalize=True) * 100, 2))

# ==========================================================
# DATA CLEANING
# ==========================================================

print("\nCleaning Dataset...")

# Replace missing values in text columns

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

for column in text_columns:
    df[column] = df[column].fillna("")

# ==========================================================
# CREATE ONE TEXT COLUMN
# ==========================================================

df["text"] = (

    df["title"] + " " +

    df["company_profile"] + " " +

    df["description"] + " " +

    df["requirements"] + " " +

    df["benefits"]

)

# ==========================================================
# TEXT CLEANING FUNCTION
# ==========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"\S+@\S+", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

print("\nPreparing Text Data...")

df["text"] = df["text"].apply(clean_text)

print("Text Cleaning Completed.")

# ==========================================================
# INPUT AND OUTPUT
# ==========================================================

X = df["text"]

y = df["fraudulent"]

print("\nDataset Ready for Vectorization!")

# ==========================================================
# TF-IDF VECTORIZATION
# ==========================================================

print("\nApplying TF-IDF Vectorization...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X = vectorizer.fit_transform(X)

print("TF-IDF Completed.")

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nNumber of Features:")
print(X.shape[1])

print("\nTF-IDF Vocabulary Size:")
print(len(vectorizer.vocabulary_))

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

print("\nPreparing Training and Testing Data...")

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples :", X_train.shape[0])

print("Testing Samples  :", X_test.shape[0])

# ==========================================================
# APPLY SMOTE
# ==========================================================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("SMOTE Applied Successfully!")

print("\nTraining Data After SMOTE:")

print(X_train.shape)

print("\nClass Distribution After SMOTE:")

print(y_train.value_counts())

print("\nData Ready for Model Training!")

# ==========================================================
# MACHINE LEARNING MODELS
# ==========================================================

print("\nTraining Machine Learning Models...")

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ),

    "Naive Bayes": MultinomialNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Support Vector Machine": LinearSVC(
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric="logloss"
    )

}

results = []

best_model = None
best_model_name = ""
best_f1 = 0

# ==========================================================
# TRAIN EACH MODEL
# ==========================================================

for name, model in models.items():

    print("\n" + "=" * 50)
    print(f"Training {name}...")
    print("=" * 50)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")

    results.append([

        name,

        accuracy,

        f1

    ])

    if f1 > best_f1:

        best_f1 = f1

        best_model = model

        best_model_name = name

print("\nAll Models Trained Successfully!")

# ==========================================================
# MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "F1 Score"
    ]
)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)

print("\nBest Model :", best_model_name)
print(f"Best F1 Score : {best_f1:.4f}")

# ==========================================================
# FINAL EVALUATION
# ==========================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

final_predictions = best_model.predict(X_test)

print(
    classification_report(
        y_test,
        final_predictions
    )
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    final_predictions
)

print(cm)

# ==========================================================
# CONFUSION MATRIX HEATMAP
# ==========================================================

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Genuine","Fake"],
    yticklabels=["Genuine","Fake"]
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ==========================================================
# MODEL COMPARISON GRAPH
# ==========================================================

plt.figure(figsize=(10,5))

sns.barplot(
    data=results_df,
    x="Model",
    y="F1 Score"
)

plt.xticks(rotation=20)

plt.title("Model Comparison (F1 Score)")

plt.tight_layout()

plt.show()

# ==========================================================
# SAVE MODEL
# ==========================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf.pkl"
)

print("\nModel Saved Successfully!")

print("Model File : models/model.pkl")

print("Vectorizer File : models/tfidf.pkl")

print("\nProject Completed Successfully!")