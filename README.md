# 🛡️ JobAegis – Fake Job Posting Detection System

## Overview

JobAegis is a Machine Learning-based web application that detects whether a job posting is **Genuine** or **Fake** using Natural Language Processing (NLP).

The system analyzes the text of a job advertisement using a TF-IDF Vectorizer and a trained XGBoost classifier.

---

## Features

- Detects Fake and Genuine Job Postings
- Machine Learning based prediction
- Natural Language Processing (NLP)
- Streamlit Web Application
- Confidence Score Display
- Easy-to-use Interface

---

## Technologies Used

- Python
- Streamlit
- Scikit-Learn
- XGBoost
- TF-IDF Vectorizer
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

---

## Machine Learning Models Tested

- Logistic Regression
- Naive Bayes
- Random Forest
- Support Vector Machine
- XGBoost (Selected Best Model)

---

## Dataset

Fake Job Postings Dataset

Source:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Total Records: **17,880**

---

## Project Structure

```
JobAegis/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── fake_job_postings.csv
│
├── models/
│   ├── model.pkl
│   └── tfidf.pkl
```

---

## Installation

Clone the repository:

```bash
git clone <repository-link>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Model Performance

Best Model: **XGBoost**

Accuracy: **98.80%**

Precision (Fake Jobs): **95%**

Recall (Fake Jobs): **79%**

F1 Score: **86.44%**

---

## Future Improvements

- BERT / RoBERTa based detection
- Explainable AI (SHAP)
- Company Reputation Verification
- Salary Anomaly Detection
- Real-time Job Portal Integration

---

## Author

**Shreyash**
