import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# Define grade mappings
train_grade_mapping = {
    'A+': 10, 'A': 9, 'A-': 8, 'B+': 7, 'B': 6, 'B-': 5,
    'C+': 4, 'C': 3, 'C-': 2, 'D': 1, 'F': 0
}

grade_mapping_for_prediction = {
    10: "A+", 9: "A", 8: "A-", 7: "B+", 6: "B", 5: "B-",
    4: "C+", 3: "C", 2: "C-", 1: "D", 0: "F"
}

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained("./grading_model")
tokenizer = AutoTokenizer.from_pretrained("./grading_model")

# Function to predict grade based on student response
def predict_grade(student_response):
    inputs = tokenizer(student_response, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
   
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
   
    # Return the grade label based on predicted class
    return grade_mapping_for_prediction.get(predicted_class, "Unknown Grade")

# Example usage
if __name__ == "__main__":
    student_response = "The Industrial Revolution was a period of major industrialization..."
    predicted_grade = predict_grade(student_response)
    print(f"Predicted Grade: {predicted_grade}")
