💳 Task 2 – Credit Card Fraud Detection

CodSoft Machine Learning Internship

📌 Project Overview

Credit card fraud is a major concern in digital transactions, causing businesses and consumers huge financial losses. This project focuses on building a machine learning model that can detect fraudulent transactions based on transaction features.

The objective is to classify a transaction as fraudulent or legitimate, using supervised learning models.

🎯 Objectives

Understand and explore a real-world fraud dataset
Handle highly imbalanced data

Build and compare multiple ML models:

Logistic Regression
Decision Tree
Random Forest
Evaluate models using robust metrics such as F1-score and ROC-AUC
Select the best-performing model for final predictions

📂 Dataset

The dataset was downloaded from Kaggle using:

kartik2112/fraud-detection

Files used:

fraudTrain.csv → training data

fraudTest.csv → final evaluation data

Target Column
is_fraud
(0 = legitimate, 1 = fraudulent)


The dataset contains:
Demographic attributes
Merchant details
Transaction amount & location
Timestamp & authentication metadata

⚠️ Note: The dataset is highly imbalanced, meaning fraudulent transactions are extremely rare. This requires careful model evaluation—accuracy alone is misleading.

🛠 Techniques & Tools Used
Feature Engineering

Dropped unique identifiers and PII-like columns

Encoded categorical features using category codes

Scaled numerical features using StandardScaler

Models Trained
Model	Notes
Logistic Regression	Performed well with class_weight='balanced'
Decision Tree	Interpretable but prone to overfitting
Random Forest	Best results with tuned hyperparameters
Metrics Evaluated
Because of class imbalance:
Precision
Recall
F1-Score (primary metric)
ROC-AUC
Confusion Matrix

🚀 Approach
1. Data Preprocessing

Loaded and explored training and test datasets
Verified class imbalance
Categorical encoding + feature scaling

2. Model Training

Trained Logistic Regression and Decision Tree directly
Optimized Random Forest for faster training using:
Reduced tree count
Controlled tree depth
Sampling optimization

3. Model Selection

The final model was chosen based on F1-score, since detecting fraud accurately matters more than raw accuracy.

📊 Results
Model	F1-Score	Notes
Logistic Regression	Good	Balanced baseline
Decision Tree	Moderate	Interpretable but noisy
Random Forest	⭐ Best	Highest F1 and ROC-AUC

The Random Forest model delivered the best performance and was used for testing on fraudTest.csv.

🧪 Example Usage
# predict_fraud(single_row)
# single_row: DataFrame with same structure as input

prediction = predict_fraud(test_sample)
print("Fraudulent" if prediction == 1 else "Legitimate")

📈 Visualizations

The notebook includes:
Confusion Matrix heatmap
ROC Curve for best model
Metric comparison table
These visualizations highlight how well the model identifies fraudulent patterns.

🎥 Video Demonstration

A demo video of this task has been uploaded on LinkedIn showcasing:
Data exploration
Model implementation
Evaluation results
Visual insights
(Link will be added here once posted)
