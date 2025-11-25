# TASK8 — K-Means Clustering for Customer Segmentation

## Structure:
- `data/` — place `Mall_Customers.csv` here (or script will generate sample data)
- `src/kmeans_clustering.py` — K-Means clustering script with elbow method
- `plots/` — output visualizations: `elbow_plot.png`, `cluster_plot.png`
- `output/` — clustered results: `clustered_output.csv`

## Dataset:
Download Mall Customers dataset from Kaggle:
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Or let the script generate sample customer data automatically.

## Usage:
```bash
cd TASK8/src
python kmeans_clustering.py
```

## Features:
- **Elbow Method**: Determines optimal number of clusters (k)
- **K-Means Clustering**: Segments customers into distinct groups
- **Visualizations**: 
  - Elbow plot showing WCSS vs number of clusters
  - Cluster plot with centroids marked
- **Output**: CSV file with cluster assignments

## Dependencies:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Results:
The script will:
1. Load or generate customer data
2. Standardize features for clustering
3. Run elbow method (k=1 to k=10)
4. Apply K-Means with optimal k (default: 5)
5. Generate visualizations
6. Save clustered output with cluster statistics
