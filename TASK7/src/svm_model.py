import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ---------- LOAD DATA ----------
data_path = "../data/breast_cancer.csv"

if os.path.exists(data_path):
    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    # Drop unnamed column if exists
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Split into X, y
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"].map({"M": 1, "B": 0})   # Convert to binary
else:
    print(f"Dataset not found at {data_path}. Using sklearn's breast cancer dataset.")
    bc = load_breast_cancer()
    X = pd.DataFrame(bc.data, columns=bc.feature_names)
    y = pd.Series(bc.target)
    # Save for future use
    df = X.copy()
    df['diagnosis'] = y.map({1: "M", 0: "B"})
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"Saved sklearn dataset to {data_path}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- LINEAR SVM ----------
linear_svm = SVC(kernel="linear", C=1)
linear_svm.fit(X_train_scaled, y_train)
y_pred_linear = linear_svm.predict(X_test_scaled)

print("\n=== LINEAR KERNEL ACCURACY ===")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# ---------- RBF SVM ----------
rbf_svm = SVC(kernel="rbf", C=1, gamma="scale")
rbf_svm.fit(X_train_scaled, y_train)
y_pred_rbf = rbf_svm.predict(X_test_scaled)

print("\n=== RBF KERNEL ACCURACY ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

# ---------- HYPERPARAMETER TUNING ----------
param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.001],
    "kernel": ["rbf"]
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train_scaled, y_train)

print("\n=== BEST PARAMETERS (GRID SEARCH) ===")
print(grid.best_params_)
print("Best Accuracy:", grid.best_score_)

# ---------- SIMPLE 2D VISUALIZATION ----------
# Pick any 2 features for decision boundary
feature_names = X.columns.tolist()
if "radius_mean" in feature_names and "texture_mean" in feature_names:
    X_2D = X[["radius_mean", "texture_mean"]]
else:
    # Use first two features if specific ones don't exist
    X_2D = X.iloc[:, :2]
    
y_2D = y

X_train_2D, X_test_2D, y_train_2D, y_test_2D = train_test_split(X_2D, y_2D, test_size=0.2, random_state=42)

scaler_2D = StandardScaler()
X_train_2D_scaled = scaler_2D.fit_transform(X_train_2D)
X_test_2D_scaled = scaler_2D.transform(X_test_2D)

# Fit models
linear_2D = SVC(kernel="linear", C=1)
rbf_2D = SVC(kernel="rbf", C=1, gamma="scale")

linear_2D.fit(X_train_2D_scaled, y_train_2D)
rbf_2D.fit(X_train_2D_scaled, y_train_2D)

# Plot function
def plot_decision_boundary(model, X, y, title, save_path):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.title(title)
    plt.xlabel("radius_mean")
    plt.ylabel("texture_mean")
    plt.savefig(save_path)
    plt.close()

plot_decision_boundary(
    linear_2D,
    X_train_2D_scaled,
    y_train_2D,
    "Linear SVM Decision Boundary",
    "../visuals/linear_decision_boundary.png"
)

plot_decision_boundary(
    rbf_2D,
    X_train_2D_scaled,
    y_train_2D,
    "RBF SVM Decision Boundary",
    "../visuals/rbf_decision_boundary.png"
)

print("\nDecision boundary images saved in /visuals/")
