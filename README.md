# 🩺 Cardiovascular Disease Risk Prediction Using Machine Learning

## 📌 Overview
Cardiovascular diseases (CVDs) are the leading cause of death globally, often developing silently until reaching critical stages. This project presents a machine learning approach to predict the likelihood of cardiovascular disease based on clinical and lifestyle features.

Using real patient data from the **Kaggle Cardiovascular Disease Dataset**, we aim to build a reliable early warning system for heart disease risk screening.

---

## 📁 Dataset
- **Source**: Kaggle - Cardiovascular Disease Dataset  
- **File Used**: `cardio_train.csv`  
- **Records**: 70,000  
- **Features**: 13 (clinical + lifestyle)  
- **Target**: `cardio` (0 = No disease, 1 = Has disease)  

### 🔑 Key Features

| Feature       | Description                                |
|---------------|--------------------------------------------|
| `age`         | Age in days (converted to years)           |
| `gender`      | 1: Female, 2: Male                         |
| `height`, `weight` | Physical metrics                    |
| `ap_hi`, `ap_lo`   | Systolic & diastolic blood pressure |
| `cholesterol` | 1: Normal, 2: Above normal, 3: Well above  |
| `gluc`        | Glucose level                              |
| `smoke`       | Smoking (0/1)                              |
| `alco`        | Alcohol intake (0/1)                       |
| `active`      | Physical activity (0/1)                    |
| `cardio`      | Target variable (0: No, 1: Yes)            |

---

## 🧠 Machine Learning Models
Implemented and compared the following models:

- Logistic Regression  
- Random Forest Classifier  
- K-Nearest Neighbors (KNN)  
- XGBoost Classifier  

### ⚙️ Optimization Techniques
- Cross-validation  
- Grid Search & Randomized Search for hyperparameter tuning  

### 🔍 Model Explainability
- **SHAP (SHapley Additive exPlanations)** to interpret model decisions  
- Feature importance analysis for key predictors

---

## 📊 Evaluation Metrics
The models were evaluated using:

- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1 Score**  
- **ROC-AUC Score**

---

## 🧪 Workflow Summary

### 🔧 Preprocessing
- Converted age from days to years  
- Normalized continuous features  
- Encoded categorical variables  

### 🏋️ Training
- Fit multiple models using train/test split and cross-validation  

### 📈 Evaluation
- Compared model performance  
- Visualized ROC curve and confusion matrix  

### 🧠 Explainability
- Used SHAP to identify and visualize key influencing features

---

## 📌 Results Summary
- The **best-performing model** (e.g., XGBoost or Random Forest) achieved **high accuracy and recall**
- **SHAP** analysis highlighted **age, blood pressure, and cholesterol** as top contributing features

---

## 🔮 Future Work
- Deploy the model as a web application using **Flask** or **Streamlit**  
- Integrate real-time health monitoring data (e.g., from wearable sensors)  
- Expand to **multi-class prediction** for different heart disease types  

---

## ⚙️ Setup

Install the required packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap jupyter

