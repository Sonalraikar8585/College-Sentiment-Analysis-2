import streamlit as st
import pickle
import numpy as np

# Load trained model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit UI
st.title("📘 College Review Sentiment Predictor")
st.write("Enter a student review to predict if it is **positive** or **negative**.")

# Text input
review = st.text_area("Enter Review:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Preprocess and predict
        review_vec = vectorizer.transform([review])
        prediction = model.predict(review_vec)
        sentiment = "Positive ✅" if prediction[0] == 1 else "Negative ❌"
        st.subheader(f"Predicted Sentiment: {sentiment}")
