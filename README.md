# Fraud Detection in Financial Transactions

## Project Overview

This project aims to detect fraudulent financial transactions using machine learning techniques. The dataset consists of financial transactions, where the goal is to predict whether a transaction is fraudulent or not. The project involves data exploration, preprocessing, feature selection, model development, and evaluation.

## Table of Contents
1. [Data Description](#data-description)
2. [Objective](#objective)
3. [Steps Involved](#steps-involved)
4. [Key Questions](#key-questions)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Development](#model-development)
7. [Model Evaluation](#model-evaluation)
8. [Conclusion](#conclusion)
9. [Future Improvements](#future-improvements)

## Data Description

The dataset contains the following columns:

- `step`: Represents a unit of time (1 hour per step).
- `type`: Type of transaction (e.g., CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER).
- `amount`: The amount of money being transacted.
- `nameOrig`: Customer ID of the sender.
- `oldbalanceOrg`: Initial balance before the transaction for the sender.
- `newbalanceOrig`: Balance after the transaction for the sender.
- `nameDest`: Customer ID of the receiver.
- `oldbalanceDest`: Initial balance before the transaction for the receiver.
- `newbalanceDest`: Balance after the transaction for the receiver.
- `isFraud`: Binary flag indicating if the transaction is fraudulent (1 for fraud, 0 for non-fraud).
- `isFlaggedFraud`: Binary flag indicating if the transaction is flagged as potentially fraudulent.

**Data Size**:
- Number of Rows: 6,362,620 rows
- Number of Columns: 10 columns

## Objective

The primary objectives of this project are:

1. To detect fraudulent transactions based on patterns in the data.
2. To build a machine learning model that classifies transactions as fraudulent (1) or non-fraudulent (0).
3. To evaluate the model and identify the most important factors contributing to fraud detection.

## Steps Involved

### Step 1: Data Cleaning
- Handling Missing Data: Ensure no missing values.
- Outliers: Detect and treat outliers in continuous variables like `amount`, `oldbalanceOrg`, and `newbalanceOrig`.
- Multi-collinearity: Check for multicollinearity among features.

### Step 2: Feature Selection
- Evaluate and select the most relevant features for the fraud detection model using correlation analysis and feature importance techniques.

### Step 3: Model Development
- **Model Selection**: Use classification models such as Logistic Regression, Decision Trees, Random Forest, or Gradient Boosting.
- **Evaluation**: Evaluate models using accuracy, precision, recall, F1-score, and AUC-ROC.
- **Cross-Validation**: Apply cross-validation for reliable model validation.

## Key Questions

### Data Cleaning:
- How were missing values, outliers, and multi-collinearity handled?

### Model Description:
- What fraud detection model was developed? Describe the model's structure and rationale.

### Feature Selection:
- What variables were selected for the model and why?

### Model Performance:
- How well does the model perform? Key metrics like accuracy, precision, and recall.

### Key Fraud Predictors:
- What are the most significant features that predict fraud?

### Business Justification:
- How do the selected features make sense from a business perspective?

### Infrastructure Improvements:
- What measures should the company implement to prevent fraud?

### Measuring Success:
- How will the company determine if fraud prevention measures are successful?

## Data Preprocessing

1. **Missing Values**: The dataset contains no missing values.
2. **Outlier Detection**: The IQR method was used to identify outliers in the `amount` column.
3. **Feature Engineering**: Only relevant features like `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest` were selected for model training.

## Model Development

The model was built using the Random Forest Classifier. The following steps were performed:

1. **Data Splitting**: Split the dataset into training and test sets (70% training, 30% testing).
2. **Model Training**: A Random Forest Classifier was trained on the training data.
3. **Feature Importance**: Feature importance was derived to understand the contribution of each feature.
4. **Model Evaluation**: The model's performance was evaluated using classification metrics and ROC-AUC.

## Model Evaluation

The model was evaluated using the following metrics:

- **Classification Report**:
    ```text
    precision    recall  f1-score   support
    0       1.00      1.00      1.00   1906351
    1       0.93      0.72      0.81      2435
    accuracy                           1.00   1908786
    macro avg       0.96      0.86      0.91   1908786
    weighted avg       1.00      1.00      1.00   1908786
    ```

- **ROC-AUC Score**: 0.9876

## Conclusion

The Random Forest model effectively detected fraudulent transactions with a high level of accuracy and precision. The key predictors of fraud include transaction amounts, balance changes, and transaction types. The project demonstrates how machine learning can be used to detect financial fraud and offers insights into key fraud indicators.

## Future Improvements

1. **Class Imbalance Handling**: Consider using techniques like SMOTE or downsampling to handle class imbalance.
2. **Feature Engineering**: Explore creating new features or transforming existing ones for better model performance.
3. **Model Optimization**: Experiment with other models like Gradient Boosting or XGBoost to compare performance.
4. **Real-Time Detection**: Implement real-time fraud detection in the production environment for quick action.

---

## Files in the Project

- `Fraud.csv`: The dataset used for training and testing.
- `fraud_detection_model.py`: Python code for data cleaning, model training, and evaluation.
- `README.md`: This file.
