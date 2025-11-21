import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap

# ---------------------------
# Create required directories
# ---------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("data/iris.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode labels — IMPORTANT FIX
le = LabelEncoder()
y = le.fit_transform(y)

# ---------------------------
# Normalize features
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------
# Evaluate multiple K-values
# ---------------------------
k_values = [1, 3, 5, 7, 9]
accuracies = []

print_output = ""

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    print_output += f"K={k} → Accuracy = {acc:.4f}\n"

best_k = k_values[np.argmax(accuracies)]
print_output += f"\nBest K = {best_k}\n"

# Final model
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
final_pred = model.predict(X_test)

cm = confusion_matrix(y_test, final_pred)
print_output += f"\nConfusion Matrix:\n{cm}\n"

# Print results to terminal
print(print_output)

# ---------------------------
# Save Accuracy Plot
# ---------------------------
plt.figure()
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy")
plt.savefig("images/k_vs_accuracy.png")
plt.close()

# ---------------------------
# Decision Boundary Plot (Using only first 2 features)
# ---------------------------
X_2d = X_scaled[:, :2]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_2d, y_train_2d)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
cmap_light = ListedColormap(['#FFDDDD', '#DDFFDD', '#DDDDFF'])
cmap_bold = ['red', 'green', 'blue']

plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)

for idx, label in enumerate(np.unique(y)):
    plt.scatter(
        X_2d[y == label, 0],
        X_2d[y == label, 1],
        c=cmap_bold[idx],
        label=le.inverse_transform([label])[0]
    )

plt.title("KNN Decision Boundary (Using 2 Features)")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend()
plt.savefig("images/decision_boundary.png")
plt.close()
