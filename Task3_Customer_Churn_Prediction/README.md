📉 Task 3 – Customer Churn Prediction

CodSoft Machine Learning Internship

📌 Project Overview

Customer churn is a key business problem for any subscription-based service. In this project, the goal is to build a machine learning model that predicts whether a customer is likely to churn (leave) or stay, based on their historical usage patterns and demographic information.

This task was completed as part of the Machine Learning Internship at CodSoft.

🎯 Objectives

Analyze a customer dataset and understand the factors influencing churn

Preprocess and clean the data (handle missing values, encode categories, scale features)

Build and compare multiple classification models such as:

Logistic Regression

Random Forest

Gradient Boosting / similar ensemble model

Evaluate models using appropriate metrics (not just accuracy)

Identify the best-performing model for churn prediction

Gain insights into which features contribute the most to churn

📂 Dataset

The dataset used is a customer churn dataset (subscription/business domain), containing information such as:

Customer demographics (e.g., age, gender, location – depending on dataset)

Subscription details (e.g., contract type, tenure, plan type)

Usage behavior (e.g., calls, internet usage, add-on services)

Target column:

Churn  (Yes / No or 1 / 0)


🔹 Dataset source: Provided via the CodSoft Task 3 instructions / Kaggle link.
🔹 Exact path and loading instructions are included in the notebook.

🛠 Tools & Techniques
Languages & Libraries

Python

Pandas, NumPy – data handling

Scikit-Learn – preprocessing, modeling, evaluation

Matplotlib / Seaborn – visualizations

Preprocessing Steps

Exploratory Data Analysis (EDA): distributions, churn rates, correlations

Handling missing values (if any)

Encoding categorical variables (e.g., One-Hot Encoding or category codes)

Feature scaling for numerical variables (e.g., StandardScaler)

Train/Validation split for model comparison

Models Explored

Logistic Regression – simple and interpretable baseline

Random Forest Classifier – tree-based ensemble model

Gradient Boosting / similar ensemble – for improved performance

🚀 Approach

Data Exploration

Checked the distribution of the target (churn vs non-churn)

Looked at relationships between features and churn (e.g., tenure, contract type, charges)

Preprocessing & Feature Engineering

Encoded categorical variables

Scaled numerical features

Removed unnecessary identifier columns if present (e.g., customer IDs)

Model Training & Comparison

Trained multiple classifiers using the processed dataset

Evaluated on a validation set using:

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

Optionally, computed ROC-AUC to measure ranking performance

Model Selection

Selected the best model based on F1-score and Recall on the churn class, since correctly identifying customers who are likely to leave is crucial from a business perspective.

📊 Evaluation

Typical metrics examined include:

Accuracy – overall correctness

Precision (Churn) – of those predicted as churn, how many actually churn

Recall (Churn) – of those who churned, how many were correctly identified

F1-Score – balance between precision and recall

Confusion Matrix – detailed view of true vs predicted labels

You can update this section later with your actual scores, for example:

Best Model: Random Forest Classifier  
Validation Accuracy : XX.XX%  
Churn Class Recall  : XX.XX%  
F1-Score (Churn)    : XX.XX  

📈 Visualizations

The notebook includes helpful plots such as:

Class distribution plot (how many customers churned vs stayed)

Correlation heatmap (if applicable)

Confusion matrix heatmap for the final model

Feature importance plot (for tree-based models like Random Forest)

These visualizations make it easier to interpret model behavior and understand drivers of churn.

🧪 How to Run

Clone the repository and navigate to the task folder:

git clone https://github.com/<your-username>/CODSOFT.git
cd CODSOFT/Task3_Customer_Churn_Prediction


Open the notebook:

In Google Colab (upload the notebook), or

In Jupyter Notebook / VS Code locally

Make sure required libraries are installed:

pip install -r requirements.txt   # if you create one

Download the dataset as per instructions in the notebook and place it in the appropriate data/ folder or path.
Run all cells in order to:
Load data

Train models

View evaluation metrics & visualizations
