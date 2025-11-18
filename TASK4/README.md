# 🎯 Logistic Regression - Binary Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![Status](https://img.shields.io/badge/Status-Ready-success.svg)

## 📝 Description

A machine learning project implementing **Logistic Regression** for binary classification tasks. This project demonstrates the complete classification workflow including data preprocessing, model training, evaluation using multiple metrics, and visualization of results through confusion matrix and ROC curve.

## 🎯 Objective

The main objective of this project is to:
- **Build a Logistic Regression classifier** for binary classification
- **Preprocess data** by handling missing values and feature scaling
- **Train and evaluate** the model using comprehensive metrics
- **Visualize model performance** using confusion matrix and ROC curve
- **Analyze feature importance** through model coefficients

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization
- **Scikit-learn** - Machine learning library

## 📁 Project Structure

```
TASK4/
│
├── data/
│   └── data.csv                      # Dataset for classification
│
├── src/
│   └── logistic_regression.py        # Main Python script
│
├── images/
│   ├── confusion_matrix.png          # Generated confusion matrix
│   └── roc_curve.png                 # Generated ROC curve
│
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Binary classification dataset (e.g., from Kaggle)

### Installation & Setup

1. **Navigate to the project directory**
   ```bash
   cd AIML_INTERN/TASK4
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
   pip install -r requirements.txt
   ```

5. **Download a dataset**
   - Download a binary classification dataset from [Kaggle](https://www.kaggle.com/)
   - Suggested datasets:
     - [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
     - [Heart Disease UCI](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
     - [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
     - [Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)
   - Place the CSV file in the `data/` folder as `data.csv`

6. **Run the script**
   ```bash
   cd src
   python logistic_regression.py
   ```

## 📊 Workflow

The script follows this comprehensive workflow:

1. **Data Loading** - Load the classification dataset from CSV
2. **Data Exploration** 
   - Display dataset shape and structure
   - Check for missing values
   - Generate statistical summary
3. **Data Preprocessing**
   - Handle missing values (median for numeric, mode for categorical)
   - Select numeric features
   - Prepare feature matrix (X) and target variable (y)
4. **Train-Test Split** - Split data (80-20) with stratification
5. **Feature Scaling** - Standardize features using StandardScaler
6. **Model Training** - Train Logistic Regression classifier
7. **Predictions** - Generate predictions and probability scores
8. **Model Evaluation** - Calculate comprehensive metrics
9. **Confusion Matrix** - Visualize true/false positives and negatives
10. **ROC Curve** - Plot ROC curve and calculate AUC score
11. **Feature Importance** - Analyze feature coefficients

## 📈 Evaluation Metrics

The model is evaluated using multiple metrics:

- **Accuracy** - Overall correctness of predictions
- **Precision** - Proportion of positive predictions that are correct
- **Recall (Sensitivity)** - Proportion of actual positives correctly identified
- **F1-Score** - Harmonic mean of precision and recall
- **ROC AUC Score** - Area under the ROC curve (model's ability to distinguish classes)
- **Confusion Matrix** - Visual breakdown of prediction results

## 📊 Generated Outputs

**Console Output:**
- Dataset information and statistics
- Class distribution
- Training progress
- Evaluation metrics (Accuracy, Precision, Recall, F1-Score)
- Classification report
- ROC AUC Score
- Feature importance (coefficients)

**Generated Images:**
- `images/confusion_matrix.png` - Heatmap showing TP, TN, FP, FN
- `images/roc_curve.png` - ROC curve with AUC score

## 🔍 Key Concepts

### Logistic Regression
A statistical method for binary classification that predicts the probability of an instance belonging to a particular class using the logistic (sigmoid) function.

### Confusion Matrix
- **True Positive (TP)**: Correctly predicted positive cases
- **True Negative (TN)**: Correctly predicted negative cases
- **False Positive (FP)**: Incorrectly predicted as positive (Type I error)
- **False Negative (FN)**: Incorrectly predicted as negative (Type II error)

### ROC Curve & AUC
- **ROC (Receiver Operating Characteristic)**: Plot of TPR vs FPR at various thresholds
- **AUC (Area Under Curve)**: Single metric summarizing model performance (0.5 = random, 1.0 = perfect)

## 👤 Author

**Kumaran**
- 📧 Email: vvkumaran24@gmail.com
- 🎓 Project: AI/ML Internship - Logistic Regression Task

---

## 📄 License

This project is open-source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

⭐ If you found this project helpful, please consider giving it a star!
