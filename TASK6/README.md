# 🎯 K-Nearest Neighbors (KNN) Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![Status](https://img.shields.io/badge/Status-Ready-success.svg)

## 📝 Description

A machine learning project implementing **K-Nearest Neighbors (KNN)** classifier for multi-class classification using the Iris dataset. This project demonstrates the KNN algorithm, optimal K value selection through cross-validation, model evaluation with comprehensive metrics, and visualization of decision boundaries.

## 🎯 Objective

The main objectives of this project are to:
- **Implement KNN classifier** for iris species classification
- **Find optimal K value** using cross-validation
- **Evaluate model performance** using accuracy, precision, recall, and F1-score
- **Visualize decision boundaries** to understand model behavior
- **Generate comprehensive visualizations** including pairplots and confusion matrix
- **Apply feature scaling** for improved KNN performance

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization
- **Scikit-learn** - Machine learning library (KNN, preprocessing, metrics)

## 📁 Project Structure

```
TASK6/
│
├── data/
│   └── iris.csv                      # Iris dataset
│
├── images/
│   ├── pairplot.png                  # Feature relationships visualization
│   ├── k_value_selection.png         # Optimal K selection plot
│   ├── confusion_matrix.png          # Confusion matrix heatmap
│   └── decision_boundary.png         # Decision boundary visualization
│
├── knn_classifier.py                 # Main Python script
├── .gitignore                        # Git ignore rules
└── README.md                         # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation & Setup

1. **Navigate to the project directory**
   ```bash
   cd AIML_INTERN/TASK6
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   
   - **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   
   - **Windows (Command Prompt):**
     ```cmd
     .venv\Scripts\activate.bat
     ```
   
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install required dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

5. **Download the dataset (optional)**
   - The script will automatically use scikit-learn's built-in Iris dataset if `iris.csv` is not found
   - Alternatively, download from [Kaggle - Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)
   - Place the CSV file in the `data/` folder as `iris.csv`

6. **Run the script**
   ```bash
   python knn_classifier.py
   ```

## 📊 Workflow

The script follows this comprehensive workflow:

1. **Setup** - Create images folder if it doesn't exist
2. **Data Loading** 
   - Try to load from `data/iris.csv`
   - If not found, load from sklearn's built-in dataset
3. **Data Exploration**
   - Display dataset structure and statistics
   - Check for missing values
   - Show target variable distribution
4. **Data Visualization**
   - Generate pairplot showing feature relationships
5. **Data Preprocessing**
   - Separate features and target
   - Train-test split (80-20) with stratification
   - Apply StandardScaler for feature scaling
6. **Optimal K Selection**
   - Test K values from 1 to 30
   - Perform 5-fold cross-validation for each K
   - Plot K vs Accuracy curve
   - Select K with highest accuracy
7. **Model Training**
   - Train KNN classifier with optimal K value
8. **Predictions & Evaluation**
   - Make predictions on test set
   - Calculate accuracy, precision, recall, F1-score
   - Display classification report
   - Perform 5-fold cross-validation
9. **Visualizations**
   - Generate confusion matrix heatmap
   - Create decision boundary plot (using first 2 features)

## 📈 Evaluation Metrics

The model is evaluated using:

- **Accuracy** - Overall correctness of predictions
- **Precision** - Proportion of positive predictions that are correct (weighted average)
- **Recall** - Proportion of actual positives correctly identified (weighted average)
- **F1-Score** - Harmonic mean of precision and recall (weighted average)
- **Cross-Validation Score** - Mean accuracy across 5 folds
- **Confusion Matrix** - Visual breakdown of predictions per class

## 📊 Generated Outputs

**Console Output:**
- Dataset shape and preview
- Dataset information and statistics
- Missing values report
- Target variable distribution
- Optimal K value with accuracy
- Model performance metrics (Accuracy, Precision, Recall, F1-Score)
- Classification report (per-class metrics)
- Cross-validation scores and mean
- Summary of generated files

**Generated Images (automatically saved in `images/` folder):**
- `pairplot.png` - Pairplot showing relationships between all features, colored by species
- `k_value_selection.png` - Line plot showing accuracy for different K values with optimal K marked
- `confusion_matrix.png` - Heatmap showing prediction results for each class
- `decision_boundary.png` - Visualization of decision boundaries using first 2 features

## 🔍 Key Concepts

### K-Nearest Neighbors (KNN)
A non-parametric, instance-based learning algorithm that classifies data points based on the majority class of their K nearest neighbors in the feature space.

**How it works:**
1. Calculate distance between test point and all training points
2. Select K nearest neighbors
3. Assign the most common class among the K neighbors

**Key Parameter:**
- **K (n_neighbors)**: Number of neighbors to consider
  - Small K: More sensitive to noise, complex boundaries
  - Large K: Smoother boundaries, may miss local patterns
  - Optimal K: Found through cross-validation

### Feature Scaling
KNN is distance-based, so feature scaling is crucial. StandardScaler is used to normalize features to have mean=0 and standard deviation=1.

### Cross-Validation
Used to find optimal K value and assess model performance. 5-fold CV splits data into 5 parts, trains on 4, tests on 1, and repeats for all combinations.

### Decision Boundary
Visual representation of how the model divides the feature space into different classes. Shows regions where the model predicts each class.

## 🎓 About the Iris Dataset

The Iris dataset is a classic dataset in machine learning:
- **150 samples** of iris flowers
- **3 species**: Setosa, Versicolor, Virginica (50 samples each)
- **4 features**: 
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)

It's an ideal dataset for classification algorithms due to its simplicity and well-separated classes.

## 👤 Author

**Kumaran**
- 📧 Email: vvkumaran24@gmail.com
- 🎓 Project: AI/ML Internship - KNN Classification Task

---

## 📄 License

This project is open-source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

⭐ If you found this project helpful, please consider giving it a star!
