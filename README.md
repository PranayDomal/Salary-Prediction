# **Salary Prediction using Machine Learning**

## **Project Overview**

This project focuses on predicting employee salaries using demographic and professional attributes. The goal is not only to achieve strong predictive performance but also to follow a disciplined, real-world machine learning workflow emphasizing interpretability, robustness, and responsible feature selection.

---

## **Dataset Description**

- Source: Kaggle
- Records: ~6,700 employees
- Target Variable: Salary
- Key Features:
  - Years of Experience
  - Education Level
  - Job Level
  - Job Category
  - Country
  - Gender

Age and race were explored during EDA but excluded from modeling due to redundancy, proxy behavior, and fairness considerations.

---

## **Modeling Approach**

1. **Data Cleaning & Feature Engineering**
  - Removed missing values and non-informative identifiers
  - Standardized education levels
  - Grouped job titles into meaningful Job Levels and Job Categories
  - Log-transformed salary to reduce skew

2. **Preprocessing Pipeline**
  - `StandardScaler` for numeric features
  - `OneHotEncoder` for categorical features
  - Implemented using `ColumnTransformer` and `Pipeline`

3. **Models Used**
  - Linear Regression (baseline, interpretability)
  - Random Forest Regressor (non-linear modeling)

---

## **Initial Model Performance Comparison (Baseline)**

| Model | R² (Actual) | MAE (Salary) |
|-----|------------|--------------|
| Linear Regression | ~0.58 | ~$22,900 |
| Random Forest | ~0.91 | ~$8,600 |

The linear model captured general trends but showed heteroscedasticity and systematic underestimation of high salaries.
Random Forest reduced typical salary prediction error by ~63%, demonstrating superior handling of non-linear relationships and feature interactions.

---

## **Business Interpretation**

- Years of Experience is the strongest salary driver.
- Job Level and Job Category provide critical structural context.
- Education adds incremental value, especially at senior roles.
- Country and gender show limited direct impact once job structure is considered.
- The model delivers realistic salary estimates suitable for workforce planning, compensation benchmarking, and HR analytics.

---

## **Project Conclusion**

- Built a robust end-to-end ML pipeline with strong EDA and validation.
- Linear Regression served as an interpretable benchmark.
- Random Forest provided substantial real-world accuracy gains.
- Residual analysis confirmed reduced bias and improved generalization.
- Model complexity was intentionally controlled to avoid overfitting.

---

## **Limitations**

- Observational dataset (no causal guarantees).
- Job levels and categories were derived heuristically.
- Random Forest offers limited interpretability compared to linear models.
- Dataset covers limited geographic regions.
- Real-world deployment would require monitoring and fairness audits.

---

## **Tools & Libraries (Dependencies)**

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## **How to Run**

1. Clone the repository:
```bash
git clone https://github.com/PranayDomal/Salary-Prediction.git
```

2. Navigate to the folder:
```bash
cd Salary-Prediction
```

3. Run the notebook:
```bash
jupyter notebook salary_prediction.ipynb
```

---

## **Project Structure**
```
├── salary_prediction.ipynb
├── salary_prediction.pdf
├── Salary_Data.csv
├── README.md
```

---

## **Author**

https://www.linkedin.com/in/pranay-domal-a641bb368/
