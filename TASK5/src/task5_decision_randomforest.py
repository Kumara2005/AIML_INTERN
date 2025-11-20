import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

# ----------------------------
# 1. MAKE IMAGES FOLDER
# ----------------------------
if not os.path.exists("../images"):
    os.makedirs("../images")

# ----------------------------
# 2. LOAD DATASET
# ----------------------------
df = pd.read_csv("../data/heart.csv")  # Load from data folder

print("Dataset Loaded Successfully!")
print(df.head())

X = df.drop("target", axis=1)
y = df["target"]

# ----------------------------
# 3. TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 4. DECISION TREE CLASSIFIER
# ----------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nDecision Tree Classification Report:\n")
print(classification_report(y_test, y_pred_dt))

# Visualize & Save Tree
plt.figure(figsize=(18, 10))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree Visualization")
plt.savefig("../images/decision_tree.png")
plt.close()

# ----------------------------
# 5. RANDOM FOREST CLASSIFIER
# ----------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:\n")
print(classification_report(y_test, y_pred_rf))

# ----------------------------
# 6. FEATURE IMPORTANCE
# ----------------------------
importances = rf.feature_importances_
feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(feat_imp)

# Save Feature Importance Plot
feat_imp.plot(kind="bar", x="Feature", y="Importance", figsize=(12, 6))
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("../images/feature_importance.png")
plt.close()

# ----------------------------
# 7. CROSS VALIDATION
# ----------------------------
cv_scores = cross_val_score(rf, X, y, cv=5)

print("\nCross Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())
