import streamlit as st
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

# Function to load the model and tokenizer from a local directory
def load_model():
    model_dir = "/home/ashok/grading_model"  # Ensure this path is correct
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory '{model_dir}' not found!")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

# Add custom CSS to style the background color and overall UI
st.markdown("""
    <style>
    body {
        background-color: #008080;  /* Teal Green */
    }
    .css-1d391kg {  /* Streamlit Title custom font size */
        font-size: 2em;
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

# Display teacher and student images
teacher_image = "path_to_teacher_image.jpg"  # Replace with actual image URL or path
student_image = "path_to_student_image.jpg"  # Replace with actual image URL or path

col1, col2 = st.columns(2)
with col1:
    st.image(teacher_image, caption="Teacher", width=150)  # Teacher image
with col2:
    st.image(student_image, caption="Student", width=150)  # Student image

# Upload CSV dataset
uploaded_file = st.file_uploader("Upload your behavioral economics dataset (.csv)", type="csv")

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
    st
