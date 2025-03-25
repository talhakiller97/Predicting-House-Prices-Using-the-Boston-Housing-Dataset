import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the provided dataset
csv_path = r"C:\Users\Talha Saeed\PycharmProjects\PythonProjectTask1\house predictionn dataset\data.csv"
df = pd.read_csv(csv_path)

# Selecting relevant features and target
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
            'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
target = 'price'

df = df[features + [target]]

# 1. Data Preprocessing
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(columns=[target]))
    y = df[target].values
    return X, y


X, y = preprocess_data(df)


# 2. Model Implementation (from scratch)
class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.theta = np.zeros(self.n)
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.theta) + self.bias
            error = y_pred - y

            self.theta -= (self.lr / self.m) * np.dot(X.T, error)
            self.bias -= (self.lr / self.m) * np.sum(error)

    def predict(self, X):
        return np.dot(X, self.theta) + self.bias


# Random Forest Regressor (from scratch)
import random


class RandomForestRegressorScratch:
    def __init__(self, n_trees=10, max_samples=0.8):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), int(len(X) * self.max_samples), replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree = LinearRegressionGD(lr=0.01, epochs=500)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)


# XGBoost (Gradient Boosting from scratch)
class XGBoostScratch:
    def __init__(self, n_estimators=50, lr=0.1):
        self.n_estimators = n_estimators
        self.lr = lr
        self.models = []

    def fit(self, X, y):
        residuals = y.copy()
        for _ in range(self.n_estimators):
            tree = LinearRegressionGD(lr=0.01, epochs=500)
            tree.fit(X, residuals)
            residuals -= self.lr * tree.predict(X)
            self.models.append(tree)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.models:
            predictions += self.lr * tree.predict(X)
        return predictions


# Train and evaluate all models
lin_reg = LinearRegressionGD(lr=0.01, epochs=2000)
lin_reg.fit(X, y)
y_pred_lr = lin_reg.predict(X)

rf_reg = RandomForestRegressorScratch(n_trees=10)
rf_reg.fit(X, y)
y_pred_rf = rf_reg.predict(X)

xgb_reg = XGBoostScratch(n_estimators=50, lr=0.1)
xgb_reg.fit(X, y)
y_pred_xgb = xgb_reg.predict(X)

# RMSE and R² evaluation
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'{model_name} RMSE: {rmse:.2f}, R²: {r2:.2f}')


evaluate_model(y, y_pred_lr, "Linear Regression")
evaluate_model(y, y_pred_rf, "Random Forest")
evaluate_model(y, y_pred_xgb, "XGBoost")

# Residual plot with separate subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

sns.histplot(y - y_pred_lr, bins=50, kde=True, color="blue", ax=axes[0])
axes[0].set_title("Linear Regression Residuals")

sns.histplot(y - y_pred_rf, bins=50, kde=True, color="green", ax=axes[1])
axes[1].set_title("Random Forest Residuals")

sns.histplot(y - y_pred_xgb, bins=50, kde=True, color="red", ax=axes[2])
axes[2].set_title("XGBoost Residuals")

plt.tight_layout()
plt.show()

# Feature Importance for Tree-based models
importances = np.abs(lin_reg.theta)
sns.barplot(x=features, y=importances)
plt.xticks(rotation=90)
plt.title('Feature Importance (Linear Regression Coefficients)')
plt.show()