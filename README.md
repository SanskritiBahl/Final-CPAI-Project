# Final-CPAI-Project
#Grading Model using Behavioral Economics Dataset
## Overview
This repository contains a machine learning model developed for predicting the grade of a student's response to behavioral economics-related concepts. The model utilizes a fine-tuned transformer model (DistilBERT) and is trained on a dataset that consists of students' responses to behavioral economics concepts and the corresponding faculty grades. The goal is to classify the responses into appropriate grade categories using Natural Language Processing (NLP) techniques.

## Objective
The primary objective of this project is to build a text classification model that can predict the grade of a student’s response based on behavioral economics concepts. The model will:

Predict grades: Given a student's response, predict the grade (A+, A, B, etc.) based on behavioral economics concepts.

Automate grading: The model can be used in an educational setting to automate the grading process, reducing the burden on educators.

Classify responses: Classify student responses to various behavioral economics concepts into appropriate grade categories.

## Methodology
1. Data Preprocessing
The dataset contains the following columns:

Student_ID: Unique identifier for each student.

Concept: The behavioral economics concept (e.g., Endowment Effect, Anchoring Bias, etc.).

Student_Response: The student's answer or explanation regarding the concept.

Faculty_Grade: The grade given by the faculty (A+, A, B, etc.).

The dataset is preprocessed in the following steps:

Renaming columns: Renamed the columns Student_Response to text and Faculty_Grade to label for consistency.

Handling missing values: Removed rows with missing values.

Grade conversion: The faculty grades were converted into numeric labels using a mapping system (e.g., A+ = 10, A = 9, etc.).

2. Tokenization and Model Setup
The dataset was tokenized using a pretrained DistilBERT model from Hugging Face's transformers library. The tokenized data was split into training and testing datasets, with an 80-20 split.

3. Model Training
The model used for classification is DistilBERT, a smaller and faster version of BERT. The training was performed using the Hugging Face Trainer API with the following parameters:

Learning rate: 2e-5

Epochs: 3

Batch size: 8

Weight decay: 0.01

4. Model Evaluation
The model was evaluated using accuracy metrics and the classification_report from sklearn to measure precision, recall, and F1 score across each grade class.

5. Grade Prediction
After training, the model can predict grades for new student responses using the following function. The student's response is tokenized and passed through the model, and the predicted grade is returned based on the highest predicted class.

6. Saving the Model
The trained model and tokenizer are saved to disk, which allows for easy deployment in production systems.

## Dataset
The dataset consists of student responses to various behavioral economics concepts and the corresponding grades given by the faculty. Each row in the dataset includes:

Student_ID: A unique identifier for each student.

Concept: A behavioral economics concept (e.g., Endowment Effect, Anchoring Bias).

Student_Response: The student's response or explanation related to the concept.

Faculty_Grade: The grade assigned by the faculty for the student's response.

## Sample of the dataset:

Student_ID	Concept	Student_Response	Faculty_Grade
S1000	Endowment Effect	People fear losses more than they value gains.	B+
S1001	Endowment Effect	Choice architecture can influence decisions.	A
S1002	Anchoring Bias	Past investments affect future decisions irrationally.	B
S1003	Hyperbolic Discounting	Risk perception changes based on framing.	C+

## Requirements
The following Python libraries are required to run this project:

transformers

datasets

torch

scikit-learn

pandas

numpy

To install the required libraries, you can use the following commands:

bash
Copy
!pip install transformers datasets torch scikit-learn pandas numpy
Usage
Clone the repository.

Install the required dependencies.

Load the dataset and run the training script.

Use the trained model to predict grades for new student responses.
