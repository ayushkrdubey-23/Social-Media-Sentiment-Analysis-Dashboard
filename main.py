import pandas as pd
import numpy as np
import re
import joblib
import os 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.train_model import *

# Load dataset
data = pd.read_csv("data/twitter.csv")

print("Dataset Loaded")
print(data.head())

# Rename columns (IMPORTANT)
data = data.rename(columns={
    "sentence": "text",
    "sentiment": "label"
})

# Clean text
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z ]", "", text)
    return text.lower()

data['cleaned'] = data['text'].apply(clean_text)

# Convert labels (0/1 → negative/positive)
data['label'] = data['label'].map({
    0: "negative",
    1: "positive"
})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Prediction
pred = model.predict(X_test_vec)

os.makedirs("outputs", exist_ok=True)

# Accuracy
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

# Classification report
report = classification_report(y_test, pred)
print("\nClassification Report:\n", report)

# Save outputs
with open("outputs/accuracy.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")

with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

print("Saved accuracy & classification report in outputs/")

# Evaluation
print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# Save model
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model saved successfully")