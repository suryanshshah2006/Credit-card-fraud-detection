# Credit-card-fraud-detection
Credit Card Fraud Detection using ANN & SMOTE

This project implements a Deep Learning solution to identify fraudulent credit card transactions. It addresses the critical challenge of extreme class imbalance in financial data, where fraudulent transactions represent only 0.17% of the total dataset.

🚀 Project Overview

Dataset: 284,807 transactions (492 frauds).

Goal: Maximize the detection of fraudulent activities (Recall) while maintaining high precision.

Architecture: Multi-layer Artificial Neural Network (ANN) built with TensorFlow/Keras.

🧠 Key Methodologies

1. Handling Extreme Imbalance

The core of this project is the comparison of sampling techniques to solve the class imbalance problem:

SMOTE (Synthetic Minority Over-sampling Technique): Synthetically generating new examples of the minority class (fraud) to balance the training set.

Under-sampling: Reducing the majority class to match the minority class size.

Over-sampling: Randomly duplicating minority class instances.

2. Data Preprocessing

Feature Scaling: Applied MinMaxScaler to normalize the 'Amount' and 'V1-V28' features.

Feature Selection: Removed the 'Time' variable to focus on transaction patterns.

📊 Results (Optimized with SMOTE)

The model achieved superior performance after balancing the data, ensuring that fraudulent transactions are not missed by the classifier:

Metric

Score

Accuracy 
95%

Precision (Fraud) 
0.98


Recall (Fraud)
0.91

F1-Score
0.95

🛠️ Tech Stack

Language: Python

Deep Learning: TensorFlow, Keras

Data Handling: Scikit-learn, Imbalanced-learn (imblearn), Pandas, Numpy

Visualization: Seaborn, Matplotlib

📈 Analysis

Through the confusion matrix and classification report, it was observed that while the standard ANN struggled with recall, the SMOTE-enhanced ANN successfully identified 91% of all fraudulent cases without significant false positives.

Developed as part of my AIML Research Portfolio.
