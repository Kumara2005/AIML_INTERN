# ============================================
# TASK 2: Exploratory Data Analysis (EDA)
# Dataset: cleaned_students_data.csv
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Better visuals
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================
# 1️⃣ Load Dataset
# ============================================
df = pd.read_csv("cleaned_students_data.csv")
print("Dataset Loaded Successfully!")
print(df.head())

# ============================================
# 2️⃣ Dataset Overview
# ============================================
print("\n--- Shape of Dataset ---")
print(df.shape)

print("\n--- Columns and Data Types ---")
print(df.dtypes)

print("\n--- Null Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# ============================================
# 3️⃣ Summary Statistics (Numeric Features)
# ============================================
print("\n--- Summary Statistics (Numeric) ---")
print(df.describe())

# ============================================
# 4️⃣ Summary of Categorical (Boolean) Variables
# ============================================
bool_cols = df.select_dtypes(include=['bool']).columns
print("\n--- Categorical/Boolean Value Counts ---")
for col in bool_cols:
    print(f"{col}:\n{df[col].value_counts()}\n")

# ============================================
# 5️⃣ Histograms for numeric columns
# ============================================
df[['math score','reading score','writing score']].hist(bins=20, color='skyblue')
plt.suptitle("Histograms of Scores")
plt.savefig('HISTOGRAM.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ HISTOGRAM.png saved")

# ============================================
# 6️⃣ Boxplots to identify distribution
# ============================================
plt.figure(figsize=(12, 5))
sns.boxplot(data=df[['math score','reading score','writing score']])
plt.title("Boxplot of Scores")
plt.savefig('BOXPLOT.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ BOXPLOT.png saved")

# ============================================
# 7️⃣ Correlation Matrix
# ============================================
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig('HEATMAP.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ HEATMAP.png saved")

# ============================================
# 8️⃣ Pairplot (numeric-only)
# ============================================
pairplot_fig = sns.pairplot(df[['math score','reading score','writing score']])
pairplot_fig.savefig('PAIRPLOT.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ PAIRPLOT.png saved")

# ============================================
# 9️⃣ Categorical vs Numeric Analysis
# ============================================

# Gender effect on scores
plt.figure(figsize=(10, 6))
sns.boxplot(x='gender_male', y='math score', data=df)
plt.title("Math Score vs Gender")
plt.savefig('CAT1.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ CAT1.png saved")

# Lunch type effect
plt.figure(figsize=(10, 6))
sns.boxplot(x='lunch_standard', y='math score', data=df)
plt.title("Math Score vs Lunch Type")
plt.savefig('CAT2.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ CAT2.png saved")

# Test preparation effect
plt.figure(figsize=(10, 6))
sns.boxplot(x='test preparation course_none', y='writing score', data=df)
plt.title("Writing Score vs Test Prep")
plt.savefig('CAT3.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ CAT3.png saved")

# ============================================
# 🔟 Insights (printed text)
# ============================================

print("\n" + "="*50)
print("All visualizations saved successfully!")
print("="*50)

print("\n--- AUTOMATIC INSIGHTS ---")

print("\n1. Mean Scores:")
print(df[['math score','reading score','writing score']].mean())

print("\n2. Which score is most correlated with math?")
print(df.corr()['math score'].sort_values(ascending=False))

print("\n3. Gender Performance Difference (math):")
print(df.groupby('gender_male')['math score'].mean())

print("\n4. Lunch Impact:")
print(df.groupby('lunch_standard')['math score'].mean())

print("\n5. Test Preparation Impact:")
print(df.groupby('test preparation course_none')['writing score'].mean())
