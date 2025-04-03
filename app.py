import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

# Function to load the model and tokenizer from a local directory
def load_model():
    model_dir = "/home/ashok/grading_model"  # Make sure this is the correct path to your model directory

    # Check if the model directory exists
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory '{model_dir}' not found! Please ensure the model is saved correctly.")

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

# Streamlit UI
st.title("Grading Prediction Model")

# Load the model and tokenizer
try:
    model, tokenizer = load_model()
    st.write("Model and tokenizer loaded successfully!")
except Exception as e:
    st.write(f"Error loading model: {e}")
    st.stop()  # Stop execution if the model couldn't be loaded

# Function to predict grade based on student response
def predict_grade(student_response):
    inputs = tokenizer(student_response, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Mapping predicted class to grade
    grade_mapping_for_prediction = {
        10: "A+", 9: "A", 8: "A-", 7: "B+", 6: "B", 5: "B-", 
        4: "C+", 3: "C", 2: "C-", 1: "D", 0: "F"
    }

    return grade_mapping_for_prediction.get(predicted_class, "Unknown Grade")

# Input field for the student response
student_response = st.text_input("Enter student response:")

if student_response:
    predicted_grade = predict_grade(student_response)
    st.write(f"Predicted Grade: {predicted_grade}")
