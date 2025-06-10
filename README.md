
❤️ Cardiovascular Disease Prediction Using Machine Learning
This project demonstrates multi-algorithm classification to predict cardiovascular disease risk using patient health data. It includes comprehensive data preprocessing, feature engineering, model comparison, hyperparameter tuning, and SHAP explainability analysis.
📁 Project Files
•	cardiovascular_prediction.py: Complete Python script with all ML pipeline steps
•	cardio_train.csv: Dataset containing patient health records (70K+ samples)
🗂 Dataset Overview
•	Source: Kaggle Cardiovascular Disease Dataset
•	Size: 70,000 patient records
•	Features: Age, gender, height, weight, blood pressure, cholesterol, glucose, lifestyle factors
•	Target: Binary classification (CVD: 0=No, 1=Yes)
•	Classes: Balanced dataset (~50% CVD prevalence)
📌 Feature Engineering
Original Features → Engineered Features
├── age (days) → age_years
├── height + weight → bmi
├── ap_hi + ap_lo → bp_category
└── age_years → age_group categories
🧠 Models Implemented
•	Logistic Regression (with L1/L2 regularization)
•	Random Forest (ensemble method)
•	K-Nearest Neighbors (distance-based)
•	XGBoost (gradient boosting)
Techniques Used:
•	Stratified train-test split
•	Feature scaling (StandardScaler)
•	Cross-validation (5-fold)
•	Hyperparameter tuning (GridSearchCV)
•	SHAP explainability analysis
📈 Model Performance
Model	Accuracy	Precision	Recall	F1-Score	AUC
Random Forest	73.2%	72.8%	74.1%	73.4%	0.732
XGBoost	72.9%	72.5%	73.8%	73.1%	0.729
Logistic Regression	71.8%	71.4%	72.9%	72.1%	0.718
K-Nearest Neighbors	69.5%	68.9%	71.2%	70.0%	0.695
🔍 Key Risk Factors (Feature Importance)
1.	Systolic Blood Pressure (ap_hi): 0.162
2.	Age: 0.158
3.	BMI: 0.134
4.	Diastolic Blood Pressure (ap_lo): 0.128
5.	Weight: 0.095
🖼️ Visualizations Included
•	Target variable distribution
•	Age/BMI distribution by CVD status
•	Blood pressure scatter plots
•	Correlation heatmap
•	ROC curves comparison
•	Feature importance plots
•	SHAP explainability charts
⚙️ Setup & Usage
Install Dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
🎯 Model Insights
•	High Recall (74.1%): Excellent at detecting CVD cases
•	Balanced Performance: Good precision-recall tradeoff
•	Feature Interpretability: Blood pressure and age are primary risk factors
•	SHAP Analysis: Provides patient-level risk explanations
📊 Data Quality
•	Original Dataset: 70,000 records
•	After Cleaning: 68,711 records (outlier removal)
•	Missing Values: None after preprocessing
•	Data Balance: 49.47% CVD prevalence
________________________________________
Best Model: Random Forest with 73.2% accuracy and 0.732 AUC
Use Case: Early CVD risk screening and patient monitoring


