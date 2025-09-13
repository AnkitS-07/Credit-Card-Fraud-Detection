# Credit-Card-Fraud-Detection

## Introduction
**Credit Card Fraud Detection System** uses machine learning to classify transactions as **genuine** or **fraudulent**. Fraud detection is crucial in financial services to protect both customers and banks from losses.

Credit Card Fraud Detection Web app :  [Credit Card Fraud Detection App](https://credit-card-fraud-detection-ankitsarkar.streamlit.app/)

---

## Workflow

###  Dataset Integration
Dataset : [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 

This dataset contains **284,807 transactions** with **30 features** (`V1–V28`, `Time`, and `Amount`) and a target variable `Class` (0 = genuine, 1 = fraud).

###  Data Preprocessing
- Checked for missing values and data types.
- Scaled `Time` and `Amount` features using `StandardScaler`.
- Handled **severe class imbalance** using **SMOTE** to synthetically generate minority class samples (fraud cases).

###  Models Used & Training
We trained multiple models to classify transactions:

- **Traditional Machine Learning Models:**
  - Logistic Regression
  - Random Forest
    
- **Boosting Models:**
  - XGBoost
  - LightGBM
  - CatBoost

All models were evaluated using **Accuracy, Precision, Recall, and F1-score**. 

---

## Results / Output

Fraud cases are extremely rare (~0.173%), making it a highly **imbalanced dataset**. Despite this challenge, our models achieved strong performance:

| Model           | Accuracy | Precision | Recall  | F1-score |
|-----------------|----------|-----------|--------|----------|
| Random Forest   | 0.99946  | 0.8454    | 0.8367 | 0.8410   |
| XGBoost         | 0.99925  | 0.7311    | 0.8878 | 0.8018   |
| CatBoost        | 0.99902  | 0.6615    | 0.8776 | 0.7544   |
| LightGBM        | 0.99823  | 0.4913    | 0.8673 | 0.6273   |
| Logistic Reg.   | 0.97421  | 0.0580    | 0.9184 | 0.1092   |

---

## Prediction
After training, the **Random Forest model** was selected as the **best-performing model**. It can predict the class of a new transaction and the probability of it being fraudulent.  
