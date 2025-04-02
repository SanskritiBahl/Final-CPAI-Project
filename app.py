import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Cache the model and tokenizer loading
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./grading_model")
    tokenizer = AutoTokenizer.from_pretrained("./grading_model")
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_model()

# Function to predict grade
def predict_grade(student_response):
    inputs = tokenizer(student_response, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    grade_mapping_for_prediction = {
        10: "A+", 9: "A", 8: "A-", 7: "B+", 6: "B", 5: "B-", 
        4: "C+", 3: "C", 2: "C-", 1: "D", 0: "F"
    }
    return grade_mapping_for_prediction.get(predicted_class, "Unknown Grade")

# Streamlit UI
st.title("Grading Prediction Model")
student_response = st.text_input("Enter student response:")

if student_response:
    predicted_grade = predict_grade(student_response)
    st.write(f"Predicted Grade: {predicted_grade}")

import os

model_dir = os.path.join(os.getcwd(), "grading_model")
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
