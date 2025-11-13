# ------------------------------
# Data Cleaning & Preprocessing
# Dataset: Students Performance
# ------------------------------

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 2. Load the dataset
# You can download it from:
# https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
df = pd.read_csv("StudentsPerformance.csv")

# 3. Basic Exploration
print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nNull Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# 4. Descriptive Statistics
print("\nSummary Statistics:\n", df.describe(include='all'))

# 5. Handle Missing Values (if any)
# (This dataset usually has no missing values, but let's demonstrate)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# 6. Encode Categorical Variables
cat_cols = df.select_dtypes(include='object').columns
print("\nCategorical Columns:", cat_cols.tolist())

# One-Hot Encoding
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("\nAfter Encoding:", df.shape)

# 7. Outlier Detection (using Boxplots)
plt.figure(figsize=(10,4))
#sns.boxplot(data=df[['math score', 'reading score', 'writing score']])
#plt.title("Boxplot for Outlier Detection")
#plt.show()

# Outlier Removal using IQR
def remove_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

# Apply to numeric columns
for column in ['math score', 'reading score', 'writing score']:
    df = remove_outliers(column)

print("Shape after removing outliers:", df.shape)

# 8. Feature Scaling (Standardization)
scaler = StandardScaler()
num_cols = ['math score', 'reading score', 'writing score']
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nAfter Scaling:\n", df.head())

# 9. Final Cleaned Data Export
df.to_csv("cleaned_students_data.csv", index=False)
print("\nCleaned dataset saved as cleaned_students_data.csv")

# 10. Final Check
print("\nFinal Columns:\n", df.columns.tolist())
print("\nFinal Shape:", df.shape)
