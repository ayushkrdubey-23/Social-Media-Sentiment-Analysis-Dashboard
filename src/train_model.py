import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Create folders
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv("data/twitter.csv")

# Rename columns
df = df.rename(columns={"sentence": "text", "sentiment": "label"})

# Clean text
df["cleaned"] = df["text"].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions
pred = model.predict(X_test_vec)

# Accuracy
acc = accuracy_score(y_test, pred)
report = classification_report(y_test, pred)

print("Accuracy:", acc)

# Save outputs
with open("outputs/accuracy.txt", "w") as f:
    f.write(str(acc))

with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

# Save predictions
pd.DataFrame({
    "text": X_test,
    "actual": y_test,
    "predicted": pred
}).to_csv("outputs/predictions.csv", index=False)

# Save model
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
