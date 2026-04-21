# Credit-Defult-Risk
Credit Defult Risk
# Enhanced Credit Risk Analysis and Modeling

This project contains an end-to-end credit risk analysis workflow built in Jupyter Notebook. It starts with raw application data, performs exploratory analysis, handles anomalies and missing values, engineers meaningful financial and behavioral features, applies feature selection, tunes multiple machine learning models, and compares their performance using several evaluation techniques.

The notebook is designed as a more polished and explainable final workflow rather than a basic baseline model. In addition to model training, it also includes threshold analysis, cross-validation, learning curves, and SHAP-based interpretation for better understanding of model behavior.

## Project Objective

The goal of this project is to predict whether a loan applicant is likely to default. The notebook focuses on binary classification, where the target variable represents repayment risk.

This workflow is especially useful for:

- academic credit risk projects
- portfolio demonstrations
- learning feature engineering for tabular financial data
- comparing tree-based models and linear models on imbalanced classification tasks

## Main Highlights

- Flexible dataset path loading for easier reuse on different machines
- Initial exploratory data analysis of train and test sets
- Class imbalance inspection
- Missing-value analysis
- Anomaly handling for employment-day outliers
- Domain-based feature engineering using income, credit, annuity, goods price, family structure, and external source variables
- Missing-flag generation for incomplete columns
- High-missing-column removal
- Median imputation for numeric fields
- Label encoding for categorical features
- Random Forest based feature selection
- Hyperparameter tuning for:
  - LightGBM
  - XGBoost
  - Logistic Regression
- Hold-out validation comparison using:
  - ROC-AUC
  - PR-AUC
  - F1 score
  - Precision
  - Recall
- Threshold optimization instead of relying only on 0.50 cutoff
- Confusion matrices at default and optimized thresholds
- Learning curves for model behavior analysis
- 5-fold cross-validation comparison
- Out-of-fold ROC-AUC check
- SHAP explainability for the strongest tree model
- Final tree-model feature importance visualization


It is built around the common Home Credit style application dataset structure, where the training data includes a `TARGET` column and the test data does not.

### Expected Target Meaning
- `TARGET = 1` → applicant has repayment difficulty / default risk
- `TARGET = 0` → applicant is not flagged as default risk

## Project Workflow

### 1. Data Loading
The notebook tries multiple possible file paths and loads whichever exists first. This makes it easier to run the notebook on different systems without rewriting too much code.

### 2. Quick Data Overview
A summary is created to inspect:

- number of rows and columns
- number of missing cells
- target class distribution

The notebook also visualizes the class imbalance and highlights columns with the highest missing percentages.

### 3. Anomaly Handling and Feature Engineering
The workflow cleans day-based variables and handles the known anomaly where `DAYS_EMPLOYED = 365243`.

A set of engineered features is then created, including:

- `CREDIT_INCOME_RATIO`
- `ANNUITY_INCOME_RATIO`
- `CREDIT_TERM`
- `GOODS_CREDIT_RATIO`
- `GOODS_INCOME_RATIO`
- `PAYMENT_RATE`
- `INCOME_PER_PERSON`
- `CREDIT_PER_PERSON`
- `INCOME_PER_CHILD`
- `YEARS_BIRTH`
- `YEARS_EMPLOYED`
- `DAYS_EMPLOYED_RATIO`
- log-transformed financial variables
- external source aggregates such as mean, min, max, standard deviation, range, and product
- document flag counts
- bureau request totals
- live flag totals

These features are added to improve predictive signal and better represent customer financial burden, stability, and external scoring patterns.

### 4. Input vs Output Visualization
The notebook compares important features against the target variable through:

- numeric distribution plots
- age-group default rate analysis
- categorical default-rate plots
- correlation heatmaps

This helps explain which variables appear more connected to credit risk.

### 5. Preprocessing
Preprocessing includes:

- creating missing-indicator flags for columns with more than 1% missing values
- dropping columns with more than 60% missing values
- filling numeric missing values with the median
- filling categorical missing values with `"missing"`
- label encoding categorical columns using combined train and test categories
- dropping constant columns
- aligning training and test feature spaces

### 6. Feature Selection
A Random Forest model is used to rank feature importance. Features contributing to the first 95% of cumulative importance are retained, which reduces noise and keeps the model focused on stronger predictors.

In this run, the selected feature count was **65**.

### 7. Hyperparameter Tuning
The notebook performs formal search-based tuning instead of manually fixed settings.

- **LightGBM** uses `GridSearchCV`
- **XGBoost** uses `RandomizedSearchCV`
- **Logistic Regression** uses `GridSearchCV` inside a scaling pipeline

This makes the final comparison more reliable than using default settings alone.

### 8. Model Training and Evaluation
Three models are trained and compared:

- LightGBM
- XGBoost
- Logistic Regression

Evaluation includes:

- ROC-AUC
- Precision-Recall AUC
- F1 score at the default threshold
- Best threshold based on F1
- Precision and recall at the optimized threshold

### 9. Threshold Analysis
Instead of assuming 0.50 is the best decision threshold, the notebook checks thresholds from 0.10 to 0.90 and identifies the threshold that produces the best balance for the selected metric.

This is particularly important in credit risk problems, where the positive class is usually rare and threshold choice strongly affects business outcomes.

### 10. Cross-Validation and Stability Checks
To avoid relying only on one train-validation split, the notebook adds:

- 5-fold ROC-AUC comparison across tuned models
- Out-of-fold ROC-AUC for the strongest tuned tree model

This gives a better view of stability and generalization.

### 11. Explainability
For the strongest tree-based model, the notebook generates:

- SHAP summary plot
- local SHAP explanation for one example case
- final feature importance ranking

This makes the model more interpretable and helps explain which variables influence predictions most strongly.

## Key Results from the Notebook

### Data Summary
- Training shape: **307,511 rows × 122 columns**
- Test shape: **307,511 rows × 122 columns**
- Positive class ratio: **8.07%**
- Final feature count after preprocessing: **191**
- Selected feature count after importance filtering: **65**

### Hyperparameter Search Summary
Best cross-validated ROC-AUC during tuning:

- **Logistic Regression:** 0.72791
- **XGBoost:** 0.71685
- **LightGBM:** 0.70145

### Hold-Out Validation Results
Validation ROC-AUC comparison:

- **XGBoost:** 0.7519
- **Logistic Regression:** 0.7480
- **LightGBM:** 0.7475

Validation PR-AUC comparison:

- **XGBoost:** 0.2412
- **Logistic Regression:** 0.2352
- **LightGBM:** 0.2278

Best-threshold F1 comparison:

- **XGBoost:** 0.3029
- **Logistic Regression:** 0.2980
- **LightGBM:** 0.2969

### Cross-Validation Results
5-fold mean ROC-AUC:

- **Logistic Regression:** 0.74431
- **XGBoost:** 0.73930
- **LightGBM:** 0.72950

Out-of-fold ROC-AUC for the strongest tuned tree model:

- **XGBoost:** 0.73885

### Explainability Insight
The SHAP and final tree-model importance outputs show that the most influential variables are dominated by external risk scores and financial burden indicators, especially:

- `EXT_SOURCE_MEAN`
- `EXT_SOURCE_MIN`
- `EXT_SOURCE_MAX`
- `EXT_SOURCE_PROD`
- `EXT_SOURCE_STD`
- `DAYS_EMPLOYED`
- `CREDIT_TERM`
- `GOODS_CREDIT_RATIO`

## Tech Stack

- Python
- Jupyter Notebook
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn
- LightGBM
- XGBoost
- SHAP

## Installation

Install the required packages before running the notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost shap jupyter
