# Predictive Modelling for Auto Insurance Premium Estimation
This project presents a machine learning pipeline to accurately estimate auto insurance premiums based on policyholder, vehicle, and regional attributes. Using a real-world dataset of over 400,000 observations, we compare multiple regression models to identify key pricing drivers and develop a robust predictive tool for insurers.

# Project Objectives

- Predict auto insurance premiums using supervised machine learning models.
- Identify significant variables influencing pricing (e.g., insured value, vehicle type).
- Compare model performance: Linear Regression vs. Random Forest vs. XGBoost.
- Leverage log transformation, encoding, and dimensionality handling for accuracy.
- Enable data-driven, risk-based pricing strategies for insurance providers.


# Techniques Used

- **Data Cleaning & Preprocessing**:
  - Missing value removal
  - Grouping high-cardinality categorical variables (e.g., vehicle make)
  - Dummy encoding of categorical features

- **EDA (Exploratory Data Analysis)**:
  - Visual analysis of log-transformed premium distributions
  - Relationship plots (insured value, claims history, etc.)
  - Correlation matrix

- **Machine Learning Models**:
  - Linear Regression (Baseline)
  - Random Forest Regressor
  - XGBoost Regressor

- **Model Evaluation**:
  - RMSE, MAE, and R-squared on test data
  - Residual analysis
  - Actual vs. predicted plots

- **Feature Importance Analysis**:
  - Top predictors identified via Random Forest (e.g., insured value, vehicle type)
