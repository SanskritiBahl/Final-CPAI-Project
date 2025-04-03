import os
import streamlit as st
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Function to load the model and tokenizer from a local directory
def load_model():
    model_dir = "/home/ashok/grading_model"  # Make sure this is the correct path to your model directory

    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory '{model_dir}' not found! Please ensure the model is saved correctly.")

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

# Add custom CSS to style the background color and overall UI
st.markdown("""
    <style>
    body {
        background-color: #008080;  /* Teal Green */
    }
    .stTextInput>div>div>input {
        background-color: #f0f0f0;  /* Light grey background for text input */
    }
    .stButton>button {
        background-color: #004d40;  /* Darker teal for button */
        color: white;
    }
    .stDataFrame {
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Grading Prediction Model")

# Upload CSV dataset
uploaded_file = st.file_uploader("Upload your behavioral economics dataset (.csv)", type="csv")

# Define the image path (Make sure the image is in the same directory as app.py)
image_path = "teacher-student.jpg"  # Replace with the correct image file name (e.g., .jpg, .png)

# Check if the image exists in the specified path
if not os.path.exists(image_path):
    st.error(f"Image not found: {image_path}")

# Display the image on Streamlit
st.image(image_path, caption="Teacher-Student Image", width=400)

# Show dataset preview if the file is uploaded
if uploaded_file is not None:
    # Load the dataset into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())  # Display first few rows of the uploaded dataset
    
    # Check if necessary columns are present in the dataset
    if "Student_Response" not in df.columns or "Faculty_Grade" not in df.columns:
        st.error("The dataset must contain 'Student_Response' and 'Faculty_Grade' columns!")
        st.stop()

# Dropdown for concept selection
concepts = [
    "Endowment Effect", "Anchoring Bias", "Hyperbolic Discounting", "Loss Aversion", 
    "Framing Effect", "Status Quo Bias", "Mental Accounting", "Sunk Cost Fallacy", 
    "Prospect Theory", "Nudging"
]

concept = st.selectbox("Select Concept", concepts)

# Input box for student response
student_response = st.text_input("Enter student response:")

# Load the model and tokenizer
try:
    model, tokenizer = load_model()
    st.write("Model and tokenizer loaded successfully!")
except Exception as e:
    st.write(f"Error loading model: {e}")
    st.stop()

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

# Button to predict grade
if student_response:
    if st.button('Predict Grade'):
        predicted_grade = predict_grade(student_response)
        st.write(f"Predicted Grade for {concept}: {predicted_grade}")

# After the dataset is uploaded, allow the user to trigger the predictions
if uploaded_file is not None and st.button('Predict Grades for All Responses'):
    st.write("Predicting grades for the entire dataset...")

    # Ensure the dataset has the necessary columns
    if "Student_Response" in df.columns:
        predictions = []
        for index, row in df.iterrows():
            response = row["Student_Response"]
            grade = predict_grade(response)
            predictions.append(grade)
        
        # Add predictions to the DataFrame
        df["Predicted_Grade"] = predictions
        
        # Show the updated dataframe with predictions
        st.write("Predictions Completed. Here is the updated dataset with predicted grades:")
        st.dataframe(df)
