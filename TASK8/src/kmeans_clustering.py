import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
import numpy as np

# -----------------------------------
# 1. Ensure folders exist
# -----------------------------------
folders = ["../plots", "../output"]

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

print("✔ Required folders are ready.\n")

# -----------------------------------
# 2. Load dataset
# -----------------------------------
data_path = "../data/Mall_Customers.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print("Dataset Loaded Successfully.\n")
else:
    print("Dataset not found. Generating sample Mall Customers data...\n")
    np.random.seed(42)
    n_samples = 200
    
    customer_id = range(1, n_samples + 1)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    age = np.random.randint(18, 70, n_samples)
    annual_income = np.random.randint(15, 140, n_samples)
    spending_score = np.random.randint(1, 100, n_samples)
    
    df = pd.DataFrame({
        'CustomerID': customer_id,
        'Gender': gender,
        'Age': age,
        'Annual Income (k$)': annual_income,
        'Spending Score (1-100)': spending_score
    })
    
    df.to_csv(data_path, index=False)
    print("Sample data generated and saved.\n")

print("First 5 Rows:\n", df.head(), "\n")

# -----------------------------------
# 3. Select Features
# -----------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
print("Selected Features:\n", X.head(), "\n")

# -----------------------------------
# 4. Scale the data
# -----------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✔ Data Scaled Successfully.\n")

# -----------------------------------
# 5. Elbow Method
# -----------------------------------
print("Calculating inertia values for K = 1 to 10...")
inertia_values = []
K = range(1, 11)

for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia_values.append(km.inertia_)

print("Inertia Values:", inertia_values, "\n")

# Save Elbow Plot
plt.figure()
plt.plot(K, inertia_values, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("../plots/elbow_plot.png")
plt.close()

print("✔ Elbow Plot Saved → ../plots/elbow_plot.png\n")

# -----------------------------------
# 6. Fit K-Means model (Optimal K = 5)
# -----------------------------------
optimal_k = 5
print(f"Training K-Means with K = {optimal_k}...\n")

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = labels
print("Cluster Labels Added:\n", df[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head(), "\n")

# -----------------------------------
# 7. Silhouette Score
# -----------------------------------
sil_score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", sil_score, "\n")

# -----------------------------------
# 8. PCA Visualization
# -----------------------------------
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(components[:, 0], components[:, 1], c=labels)
plt.title("K-Means Clusters (PCA Visualization)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("../plots/cluster_plot.png")
plt.close()

print("✔ Cluster Plot Saved → ../plots/cluster_plot.png\n")

# -----------------------------------
# 9. Save Final Output CSV
# -----------------------------------
df.to_csv("../output/clustered_output.csv", index=False)
print("✔ Clustered Dataset Saved → ../output/clustered_output.csv\n")

print("🎉 Task 8 Completed Successfully!")
