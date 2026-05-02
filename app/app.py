import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import re
from collections import Counter
import os

# For evaluation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Create images folder
os.makedirs("images", exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("Social Media Sentiment Analysis Dashboard")

# -----------------------------
# SINGLE PREDICTION
# -----------------------------
st.header("Single Text Prediction")

text_input = st.text_area("Enter text")

if st.button("Predict"):
    if text_input.strip():
        cleaned = clean_text(text_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        st.write("Prediction:", pred)
    else:
        st.warning("Enter text first")

# -----------------------------
# FILE UPLOAD
# -----------------------------
st.header("Bulk Prediction using CSV")

file = st.file_uploader("Upload twitter.csv file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("Raw Data Preview")
    st.write(df.head())

    # -----------------------------
    # COLUMN HANDLING
    # -----------------------------
    if "sentence" in df.columns:
        text_col = "sentence"
    elif "text" in df.columns:
        text_col = "text"
    else:
        st.error("CSV must contain 'sentence' or 'text' column")
        st.stop()

    # -----------------------------
    # CLEAN TEXT
    # -----------------------------
    df["cleaned"] = df[text_col].apply(clean_text)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    df["prediction"] = model.predict(vectorizer.transform(df["cleaned"]))

    st.success("Prediction completed")

    st.subheader("Prediction Sample")
    st.write(df[[text_col, "prediction"]].head())

    # -----------------------------
    # PIE CHART
    # -----------------------------
    st.subheader("Sentiment Distribution (Pie Chart)")
    fig1 = px.pie(df, names="prediction")
    st.plotly_chart(fig1)
    try:
       fig1.write_image("images/sentiment_pie.png")
    except Exception as e:
       st.warning(f"Pie chart not saved: {e}")
    fig1.write_image("images/sentiment_pie.png")

    # -----------------------------
    # BAR CHART
    # -----------------------------
    st.subheader("Sentiment Count (Bar Chart)")

    count_df = df["prediction"].value_counts().reset_index()
    count_df.columns = ["Sentiment", "Count"]

    fig2 = px.bar(count_df, x="Sentiment", y="Count", text="Count")
    st.plotly_chart(fig2)
    try:
       fig2.write_image("images/sentiment_bar.png")
    except Exception as e:
        st.warning(f"Bar chart not saved: {e}")
    fig2.write_image("images/sentiment_bar.png")

    # -----------------------------
    # WORD FREQUENCY
    # -----------------------------
    st.subheader("Most Common Words")

    words = " ".join(df["cleaned"]).split()
    freq = Counter(words).most_common(10)

    word_df = pd.DataFrame(freq, columns=["Word", "Frequency"])

    fig3 = px.bar(word_df, x="Word", y="Frequency", text="Frequency")
    st.plotly_chart(fig3)
    try:
       fig3.write_image("images/word_freq.png")
    except Exception as e:
     st.warning(f"Word graph not saved: {e}")
    fig3.write_image("images/word_freq.png")

    # -----------------------------
    # MODEL EVALUATION (ONLY IF LABEL EXISTS)
    # -----------------------------
    if "sentiment" in df.columns:

        st.subheader("Model Evaluation")

        y_true = df["sentiment"]
        y_pred = df["prediction"]

        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        st.write("Accuracy:", round(acc, 4))

        # Classification report
        report = classification_report(y_true, y_pred)
        st.text("Classification Report")
        st.text(report)

        # Confusion matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # SAVE IMAGE
        fig_cm.savefig("images/confusion_matrix.png")
        st.pyplot(fig_cm)

    # -----------------------------
    # DOWNLOAD RESULTS
    # -----------------------------
    csv = df.to_csv(index=False)

    st.download_button(
        "Download Predictions CSV",
        csv,
        "sentiment_results.csv",
        "text/csv"
    )

    # -----------------------------
    # FINAL TABLE
    # -----------------------------
    st.subheader("Final Data with Predictions")
    st.dataframe(df)


