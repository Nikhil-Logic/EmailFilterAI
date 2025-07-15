import streamlit as st
import joblib

model = joblib.load('spam.joblib')

# Streamlit UI
st.title("SMS Spam Classifier")

user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    prediction = model.predict([user_input])
    label = "Spam" if prediction[0] == 1 else "Ham"
    st.write(f"Prediction: **{label}**")