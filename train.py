"""
House Price Predictor - California Housing Dataset
Run: python train.py
Outputs: correlation_heatmap.png, model.pkl, scaler.pkl, metrics printed to console
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------- 1. Load data ----------
data = fetch_california_housing(as_frame=True)
df = data.frame  # target column = 'MedHouseVal'
print("Shape:", df.shape)
print(df.isnull().sum())  # this dataset has no NaNs, but always check

# ---------- 2. Feature engineering ----------
# Ratios often beat raw counts for regression
df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]
df["BedroomsRatio"] = df["AveBedrms"] / df["AveRooms"]
df["PopPerHousehold"] = df["Population"] / df["AveOccup"]

# ---------- 3. EDA ----------
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()
print("Saved correlation_heatmap.png")

# ---------- 4. Train/test split ----------
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 5. Baseline: Linear Regression ----------
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
print(f"\nLinear Regression -> RMSE: {lr_rmse:.4f}, R2: {r2_score(y_test, lr_pred):.4f}")

# ---------- 6. Gradient Boosting (usually 20-30% better RMSE) ----------
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train_scaled, y_train)
gb_pred = gb.predict(X_test_scaled)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
print(f"Gradient Boosting  -> RMSE: {gb_rmse:.4f}, R2: {r2_score(y_test, gb_pred):.4f}")

# ---------- 7. Quick hyperparameter tuning (optional, takes ~1-2 min) ----------
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4],
    "learning_rate": [0.05, 0.1],
}
grid = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3,
                     scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(X_train_scaled, y_train)
best_gb = grid.best_estimator_
best_pred = best_gb.predict(X_test_scaled)
best_rmse = np.sqrt(mean_squared_error(y_test, best_pred))
print(f"Tuned GB           -> RMSE: {best_rmse:.4f}, Best params: {grid.best_params_}")

# ---------- 8. Feature importance ----------
importances = pd.Series(best_gb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop features:\n", importances.head())

# ---------- 9. Save best model + scaler for the Streamlit demo ----------
joblib.dump(best_gb, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "features.pkl")
print("\nSaved model.pkl, scaler.pkl, features.pkl")