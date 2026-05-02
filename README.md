#  Social Media Sentiment Analysis Dashboard

---

##  Overview

The **Social Media Sentiment Analysis Dashboard** is a Machine Learning-based project that analyzes textual data from social media and classifies user sentiment as **Positive** or **Negative**.

This project demonstrates how organizations can automatically understand public opinion at scale and make data-driven decisions using Artificial Intelligence.

---

##  Problem Statement

In today’s digital era, millions of users express their opinions on platforms like social media, product reviews, and online forums.

However:

* The data is **huge in volume**
* It is **unstructured (text format)**
* Manual analysis is **time-consuming and inefficient**

Businesses struggle to:

* Understand customer satisfaction
* Detect negative feedback early
* Monitor brand reputation
* Analyze product reviews at scale

---

##  Solution

This project provides an **automated sentiment analysis system** that:

* Processes raw text data
* Cleans and prepares it for analysis
* Converts text into numerical features
* Uses Machine Learning to classify sentiment
* Displays insights through an interactive dashboard

---

##  How It Works (End-to-End Pipeline)

```
Raw Social Media Text
        ↓
Text Cleaning & Preprocessing
        ↓
TF-IDF Feature Extraction
        ↓
Machine Learning Model (Logistic Regression)
        ↓
Sentiment Prediction (Positive / Negative)
        ↓
Streamlit Dashboard Visualization
```

---

##  Tech Stack

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Joblib
* **NLP Technique:** TF-IDF Vectorization
* **Machine Learning Model:** Logistic Regression
* **Visualization:** Plotly
* **Dashboard:** Streamlit

---

##  Project Structure

```
Social-Media-Sentiment-Analysis-Dashboard/
│
├── data/                # Dataset
├── models/              # Saved ML model & vectorizer
├── src/                 # Modular ML pipeline code
├── app/                 # Streamlit dashboard
├── outputs/             # Results (accuracy, predictions)
├── docs/                # Project notes
├── images/              # Screenshots
│
├── main.py              # Model training script
├── requirements.txt     # Dependencies
└── README.md
```

---

##  Installation & Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python main.py
```

### Step 3: Launch Dashboard

```bash
streamlit run app/app.py
```

### Step 4: Open in Browser

```
http://localhost:8501
```

---

##  Features

* Real-time sentiment prediction
* Text input analysis
* CSV file upload support
* Interactive dashboard
* Visualization using pie charts

---

##  Outputs

The project generates the following outputs:

###  Model Performance

* Accuracy Score
* Precision, Recall, F1-score

###  Files Generated (in outputs/)

* `accuracy.txt` → Model accuracy
* `classification_report.txt` → Detailed evaluation
* `predictions.csv` → Predicted results

###  Dashboard Output

* Sentiment prediction for input text
* Pie chart showing sentiment distribution
* Dataset preview

---

##  Screenshots

(these images inside `images/` folder)

* Dataset preview
* Model accuracy output
* Dashboard interface
* Sentiment distribution chart
* Prediction result

---

##  Real-World Applications

* Customer feedback analysis
* Brand sentiment monitoring
* Product review classification
* Social media analytics
* Market research

Used in industries like:

* E-commerce (Amazon, Flipkart)
* Food delivery (Zomato, Swiggy)
* Streaming platforms (Netflix)
* Banking & fintech

---

##  Learning Outcomes

* Understanding of NLP preprocessing
* Feature extraction using TF-IDF
* Machine Learning model building
* Model evaluation techniques
* Dashboard development using Streamlit
* End-to-end project structuring

---

##  Future Improvements

* Add Neutral sentiment classification
* Use advanced NLP models (BERT, Transformers)
* Deploy dashboard online
* Integrate real-time APIs (Twitter/YouTube)
* Add word cloud visualization

---

##  Acknowledgement

This project was completed under the guidance of
**Mr. Umesh Yadav**

As part of the **IIP Program in collaboration with EDC IIT Delhi**

---

##  Author

**Ayush Kumar Dubey**

---

##  Project Note

This is my **3rd Machine Learning Project out of a 5-project series**, focused on building industry-level, real-world applications and strengthening my Data Science portfolio.

---

##  Conclusion

This project successfully demonstrates how Machine Learning and NLP can be used to transform unstructured text into meaningful insights. It provides a practical and scalable solution for sentiment analysis, similar to systems used in real-world business environments.

---
