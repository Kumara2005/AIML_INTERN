# ============================================
#   TASK 3 — LINEAR REGRESSION (Python Script)
#   Works with housing.csv
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------
df = pd.read_csv("data/housing.csv")

print("\nLoaded Dataset:")
print(df.head())

# --------------------------------------------
# 2. SIMPLE LINEAR REGRESSION
# --------------------------------------------
X_simple = df[['Area Population']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

y_pred_simple = model_simple.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred_simple)
mse = mean_squared_error(y_test, y_pred_simple)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_simple)

print("\n===== SIMPLE LINEAR REGRESSION =====")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²   :", r2)

# Plot and Save Instead of Showing
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color='orange', label='Actual', alpha=0.6)
plt.plot(X_test, y_pred_simple, color='black', linewidth=2, label='Predicted')
plt.xlabel("Area Population")
plt.ylabel("Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)

# Save plot into the folder
plt.savefig("simple_linear_regression_plot.png")

# Close plot so program continues
plt.close()

# --------------------------------------------
# 3. MULTIPLE LINEAR REGRESSION
# --------------------------------------------
X_multi = df.drop(columns=['Price', 'Address'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred_multi = model_multi.predict(X_test)

# Evaluation
mae_multi = mean_absolute_error(y_test, y_pred_multi)
mse_multi = mean_squared_error(y_test, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)
r2_multi = r2_score(y_test, y_pred_multi)

print("\n===== MULTIPLE LINEAR REGRESSION =====")
print("MAE :", mae_multi)
print("MSE :", mse_multi)
print("RMSE:", rmse_multi)
print("R²   :", r2_multi)

# Coefficients
coeff_df = pd.DataFrame({
    "Feature": X_multi.columns,
    "Coefficient": model_multi.coef_
})

print("\nMODEL COEFFICIENTS:")
print(coeff_df)

# --------------------------------------------
# 4. Predict New Sample
# --------------------------------------------
sample = pd.DataFrame({
    "Avg. Area Income": [60000],
    "Avg. Area House Age": [7],
    "Avg. Area Number of Rooms": [6],
    "Avg. Area Number of Bedrooms": [3],
    "Area Population": [25000]
})

prediction = model_multi.predict(sample)
print("\nPredicted Price for Sample House:", prediction[0])
