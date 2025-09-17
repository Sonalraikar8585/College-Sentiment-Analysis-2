
---

## **Project 2: College Review Sentiment Predictor (By Review Text)**

```markdown
# College Review Sentiment Predictor – By Review Text

## Project Description
This project predicts the sentiment of individual college reviews provided by students or faculty. Users can submit a textual review, and the system predicts whether the sentiment is **Positive** or **Neutral**.  

This project is useful for analyzing feedback in real-time and understanding public perception of a college based on textual reviews.

---

## Features
- Predict sentiment of an individual review as **Positive** or **Neutral**.
- Simple interface for entering textual feedback.
- Supports real-time review sentiment analysis.
- Can be extended to include more classes or advanced NLP models.

---

## Technologies Used
- **Programming Language:** Python  
- **Libraries:** scikit-learn, pandas, numpy, nltk (for NLP preprocessing), joblib/pickle (for saving models)  
- **Model:** Machine Learning Classifier (e.g., Logistic Regression, Naive Bayes)

---

## How to Use

1.Clone the repository: 2.Create a virtual environment: python -m venv venv 3.Activate the virtual environment: .\venv\Scripts\activate 4.Install required Python packages: pip install pandas numpy scikit-learn matplotlib seaborn streamlit 5.Train the machine learning model: python train_model.py 6.Run the Streamlit web app: streamlit run app.py


