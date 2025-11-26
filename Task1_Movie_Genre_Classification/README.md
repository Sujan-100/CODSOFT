📌 Project Overview

This project focuses on building a movie genre classification system using textual information from movie plot summaries. The task is to predict a movie’s genre based only on its plot — an application of Natural Language Processing (NLP) and machine learning.

This task was completed as part of the CodSoft Machine Learning Internship.

🎯 Objective

Convert movie plot summaries into numerical features using TF-IDF

Train multiple ML models for classification:

Logistic Regression
Naive Bayes
Linear SVM
Compare model performance and select the best classifier
Evaluate the chosen model on a separate test dataset
Visualize insights such as confusion matrix and top genre-related features

📂 Dataset

The dataset was downloaded from Kaggle using:

hijest/genre-classification-dataset-imdb


Files used:

train_data.txt – full training dataset
test_data.txt – test set without labels
test_data_solution.txt – true labels for test set

Each row includes:

title ::: genre ::: plot

🛠 Techniques & Tools Used
NLP Processing
TF-IDF Vectorization (unigrams + bigrams)
Stopword removal (English)
Text cleaning and normalization

ML Models

Logistic Regression (best performer)
Multinomial Naive Bayes
Linear Support Vector Machine (SVM)

Evaluation Metrics

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Classification Report

Libraries

Python
Scikit-Learn
Pandas
NumPy
Matplotlib
Seaborn

🚀 Approach
1. Load and explore the dataset

Parsed the .txt files using custom delimiters

Inspected label distribution

Checked for missing or corrupt rows

2. Preprocessing

Extracted plot as text input

Extracted genre as the target

Converted text into TF-IDF vectors

Split training set → 80% train / 20% validation

3. Model Training

Trained and evaluated the following models:

Model	Validation Accuracy
Logistic Regression	⭐ Best
Linear SVM	Good
Naive Bayes	Moderate
4. Final Evaluation

The best model (Logistic Regression or SVM depending on run) was tested on the official test_data.txt using the true labels from test_data_solution.txt.

📊 Visualizations
Confusion Matrix

A large heatmap displaying the model’s performance for each genre class.

Top Features per Genre

Bar plots showing the most important TF-IDF words that influence genre classification — e.g.,

“haunted”, “ghost”, “murder” → Horror

“love”, “relationship” → Romance

“mission”, “agent” → Action

🧪 How to Run This Project

Place dataset files in a folder:

/content/data/


Run the notebook in Google Colab.
Install necessary packages (scikit-learn, seaborn).
Execute the training and evaluation code blocks.
Explore visualizations and predictions.

🔮 Prediction Example
example_plot = """
A group of astronauts embark on a journey through a wormhole to save humanity.
"""
predict_genre(example_plot)

🏆 Results

The final model achieved:
High accuracy across major genres
Strong generalization on unseen test data
Clear genre-based keyword patterns extracted from TF-IDF

📎 GitHub Repository

Complete source code and notebooks are available here:
https://github.com/Sujan-100
