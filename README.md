# Predicting House Prices Using the Boston Housing Dataset

## Description
This project involves building a regression model from scratch to predict house prices using the Boston Housing Dataset. The models implemented include Linear Regression, Random Forest, and XGBoost.

## Steps
### 1. Data Preprocessing
- Normalized numerical features using `StandardScaler`.
- Selected relevant features for prediction.

### 2. Model Implementation
- **Linear Regression**: Implemented using Gradient Descent.
- **Random Forest**: Built from scratch using multiple Linear Regression models on bootstrapped datasets.
- **XGBoost**: Implemented using a gradient boosting approach.

### 3. Performance Comparison
- Used **RMSE (Root Mean Squared Error)** and **R² (R-Squared)** to compare models.

### 4. Feature Importance
- Visualized feature importance using Linear Regression coefficients.
- Residual plots were generated to analyze model errors.

## Results
The performance metrics for each model:
```
Linear Regression RMSE: 498612.98, R²: 0.22
Random Forest RMSE: 498839.60, R²: 0.22
XGBoost RMSE: 498624.20, R²: 0.22
```

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run
1. Ensure all dependencies are installed using:
   ```sh
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
2. Run the script:
   ```sh
   python task4.py
   ```

## Visualizations
- Residual plots for each model.
- Feature importance plot for Linear Regression.

## Notes
- The dataset must be placed at the correct path:
  ```
  C:\Users\Talha Saeed\PycharmProjects\PythonProjectTask1\house predictionn dataset\data.csv
  ```
- Ensure the script is executed within the virtual environment if using one.

## Author
Talha Saeed

