# 🌳 Decision Tree & Random Forest Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📝 Description

A machine learning project implementing **Decision Tree** and **Random Forest** classifiers for heart disease prediction. This project demonstrates tree-based classification algorithms, model comparison using accuracy metrics, feature importance analysis, and decision tree visualization.

## 🎯 Objective

The main objectives of this project are to:
- **Build Decision Tree classifier** for heart disease prediction
- **Build Random Forest classifier** with 200 estimators
- **Compare model performance** using accuracy and classification reports
- **Visualize decision tree structure** for interpretability
- **Analyze feature importance** to identify key heart disease predictors
- **Apply 5-fold cross-validation** for robust performance estimation

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **Scikit-learn** - Machine learning library (Tree models, metrics, cross-validation)
- **OS** - File and directory operations

## 📁 Project Structure

```
TASK5/
│
├── data/
│   └── heart.csv                          # Heart disease dataset
│
├── images/
│   ├── decision_tree.png                  # Generated decision tree visualization
│   └── feature_importance.png             # Generated feature importance plot
│
├── src/
│   └── task5_decision_randomforest.py     # Main Python script
│
├── .gitignore                             # Git ignore rules
└── README.md                              # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Heart disease dataset from Kaggle

### Installation & Setup

1. **Navigate to the project directory**
   ```bash
   cd AIML_INTERN/TASK5
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
   pip install pandas matplotlib scikit-learn
   ```

5. **Download the dataset**
   - Download the Heart Disease dataset from Kaggle:
     - [Heart Disease UCI](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
     - [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
   - Place the CSV file in the `data/` folder as `heart.csv`

6. **Run the script**
   ```bash
   cd src
   python task5_decision_randomforest.py
   ```

## 📊 Workflow

The script follows this workflow:

1. **Setup** - Create images folder if it doesn't exist
2. **Data Loading** - Load heart disease dataset from `data/heart.csv`
3. **Data Preparation**
   - Separate features (X) and target (y)
   - Display first 5 rows of dataset
4. **Train-Test Split** - Split data into 80% training and 20% testing
5. **Decision Tree Training**
   - Train DecisionTreeClassifier with random_state=42
   - Make predictions on test set
   - Display accuracy and classification report
   - Visualize tree structure and save to `images/decision_tree.png`
6. **Random Forest Training**
   - Train RandomForestClassifier with 200 estimators
   - Make predictions on test set
   - Display accuracy and classification report
7. **Feature Importance Analysis**
   - Extract feature importances from Random Forest
   - Display ranked features by importance
   - Generate bar plot and save to `images/feature_importance.png`
8. **Cross-Validation**
   - Perform 5-fold cross-validation on Random Forest
   - Display individual CV scores and mean score

## 📈 Evaluation Metrics

Both models are evaluated using:

- **Accuracy** - Overall correctness of predictions
- **Classification Report** - Detailed precision, recall, and F1-score for each class
- **Feature Importance** - Ranking of features by their predictive power (Random Forest only)
- **Cross-Validation Score** - Mean accuracy across 5 folds (Random Forest only)

## 📊 Generated Outputs

**Console Output:**
- Dataset preview (first 5 rows)
- Decision Tree accuracy score
- Decision Tree classification report (precision, recall, F1-score per class)
- Random Forest accuracy score
- Random Forest classification report
- Feature importance rankings (all features sorted by importance)
- Cross-validation scores (5 individual scores + mean)

**Generated Images:**
- `images/decision_tree.png` - Visual representation of the decision tree structure with feature names and class labels
- `images/feature_importance.png` - Bar chart showing feature importance from Random Forest

## 🔍 Key Concepts

### Decision Tree
A tree-structured classifier that makes decisions by splitting data based on feature values. Easy to interpret and visualize but prone to overfitting.

**Configuration:**
- `random_state=42`: For reproducible results
- Default hyperparameters used for simplicity

### Random Forest
An ensemble method that builds multiple decision trees and combines their predictions through voting. More robust and less prone to overfitting than single decision trees.

**Configuration:**
- `n_estimators=200`: 200 trees in the forest for robust predictions
- `random_state=42`: For reproducible results
- Default hyperparameters for other settings

### Feature Importance
Measures how much each feature contributes to the model's predictions. Higher values indicate more important features for classification.

### Cross-Validation
Technique to assess model performance by training on different subsets of data, providing a more reliable performance estimate.

## 🏆 Model Comparison

| Aspect | Decision Tree | Random Forest |
|--------|--------------|---------------|
| **Interpretability** | High (easy to visualize) | Low (multiple trees) |
| **Overfitting Risk** | High | Low (ensemble effect) |
| **Training Speed** | Fast | Slower (multiple trees) |
| **Accuracy** | Good | Better (typically) |
| **Stability** | Less stable | More stable |

## 👤 Author

**Kumaran**
- 📧 Email: vvkumaran24@gmail.com
- 🎓 Project: AI/ML Internship - Decision Tree & Random Forest Task

---

## 📄 License

This project is open-source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

⭐ If you found this project helpful, please consider giving it a star!
