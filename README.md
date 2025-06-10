# 🩺 Cardiovascular Disease Risk Prediction Using Machine Learning
## 📌 Overview
Cardiovascular diseases (CVDs) are the leading cause of death globally, often developing silently until reaching critical stages. This project presents a machine learning approach to predict the likelihood of cardiovascular disease based on clinical and lifestyle features. By using real patient data from the Kaggle Cardiovascular Disease dataset, we aim to build a reliable early warning system for heart disease risk screening.

---

## 📁 Dataset
- **Source:** [Kaggle - Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **File used:** `cardio_train.csv`
- **Records:** 70,000
- **Features:** 13 (clinical + lifestyle)
- **Target:** `cardio` (0 = no disease, 1 = has disease)

### Key Features:
| Feature       | Description                               |
|---------------|-------------------------------------------|
| age           | Age in days (converted to years)          |
| gender        | 1: Female, 2: Male                        |
| height, weight| Physical metrics                          |
| ap_hi, ap_lo  | Blood pressure (systolic, diastolic)      |
| cholesterol   | 1: Normal, 2: Above normal, 3: Well above |
| gluc          | Glucose level                             |
| smoke, alco   | Smoking & alcohol intake (0/1)            |
| active        | Physical activity (0/1)                   |
| cardio        | Target variable (0: No, 1: Yes)           |

---

## 🧠 Machine Learning Models
Implemented and compared:
- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- XGBoost Classifier

### Optimization Techniques:
- Cross-validation
- Grid Search & Randomized Search for hyperparameter tuning

### Model Explainability:
- **SHAP** (SHapley Additive exPlanations) for interpreting model decisions
- Feature importance analysis for insights into key predictors

---

## 📊 Evaluation Metrics
Used the following metrics for model evaluation:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

---

## 🧪 Workflow Summary
1. **Preprocessing:**
   - Convert age from days to years
   - Normalize continuous features
   - Encode categorical variables

2. **Training:**
   - Fit multiple models using train/test split and cross-validation

3. **Evaluation:**
   - Compare performance across models
   - Visualize ROC curve and confusion matrix

4. **Explainability:**
   - Use SHAP to interpret key features affecting predictions

## 📌 Results Summary
- The best-performing model (e.g., XGBoost or Random Forest) achieved **high accuracy and recall**.
- SHAP plots revealed that **age, blood pressure, and cholesterol** were top contributing features.

---
## 🔮 Future Work

- Deploy the model as a web application using Flask or Streamlit
- Integrate real-time health monitoring data (e.g., wearable sensors)
- Expand to multi-class prediction for different heart disease types

---
##⚙️ Setup Install required packages:
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pillow
