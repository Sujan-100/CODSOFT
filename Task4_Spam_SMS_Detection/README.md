📩 Task 4 – Spam SMS Detection

CodSoft Machine Learning Internship

📌 Project Overview

The goal of this project is to build a machine learning model that classifies SMS messages as either Spam or Ham (Not Spam). This helps prevent unwanted messages, phishing attempts, promotions, and fraudulent activities by automatically filtering them.

This task was completed as part of the Machine Learning Internship at CodSoft.

🎯 Objectives

Preprocess raw SMS text to make it machine-readable

Extract meaningful text features using TF-IDF Vectorization

Train and compare multiple machine learning models such as:

Logistic Regression

Naive Bayes (especially suited for text data)

Support Vector Machine (SVM)

Evaluate the performance of models using classification metrics

Predict whether a given SMS message is spam or legitimate

📂 Dataset

The dataset consists of SMS messages labeled as:

ham  → normal/legitimate message
spam → unsolicited message


Each row contains:

Column	Description
label	Spam or Ham
message	SMS content

Dataset was provided in the CodSoft instructions / Kaggle source (SMS Spam Collection Dataset).

🛠 Tech Stack Used
Libraries

Python

Pandas, NumPy — data handling

Scikit-Learn — ML models and evaluation

Matplotlib / Seaborn — visualizations

Techniques

Text cleaning (lowercase conversion, removing punctuation, etc.)

TF-IDF vectorization for converting text to numeric features

Model comparison using various classification algorithms

Confusion matrix and performance reports

🚀 Approach

Data Loading & Inspection

Checked shape, missing data, label distribution

Text Preprocessing

Lowercased text

Removed punctuation and special characters

Removed stopwords where applicable

Converted processed text into TF-IDF vectors

Model Training

Trained multiple models:

Naive Bayes (performed best for spam detection)

Logistic Regression

Support Vector Machine

Model Evaluation

Compared models using:

Accuracy

Precision

Recall

F1-score

Visualized confusion matrices for better understanding

Prediction

Tested the best model on new SMS text inputs

📊 Results
Model	Performance
Naive Bayes	⭐ Best — high precision and recall
Logistic Regression	Good baseline
SVM	Competitive but slower

The Naive Bayes model performed exceptionally well due to its probability-based nature, making it ideal for text classification.

🧪 Usage Example
msg = ["Congratulations! You have won a free gift!"]
prediction = model.predict(tfidf.transform(msg))
print(prediction)

📈 Visualizations Included

Confusion Matrix Heatmap

Model performance comparison

Label distribution chart

These help understand where the model performs well and where improvements may be needed.
