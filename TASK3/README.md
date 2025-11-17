# 📊 Linear Regression - Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📝 Description

A machine learning project implementing **Simple and Multiple Linear Regression** to predict housing prices based on various features. This project demonstrates the complete ML workflow including data loading, model training with both single and multiple features, performance evaluation using key metrics, and visualization of results.

## 🎯 Objective

The main objective of this project is to:
- **Build Simple Linear Regression** model using a single feature (Area Population)
- **Build Multiple Linear Regression** model using multiple features
- **Compare performance** between simple and multiple regression models
- **Evaluate model performance** using metrics (MAE, MSE, RMSE, R²)
- **Visualize predictions** vs actual values
- **Make predictions** on new sample data

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Scikit-learn** - Machine learning library (Linear Regression, train-test split, metrics)

## 📁 Project Structure

```
TASK3/
│
├── data/
│   └── housing.csv                       # Housing dataset (USA Housing)
│
├── linear_regression.py                  # Main Python script
├── simple_linear_regression_plot.png     # Generated visualization
├── requirements.txt                      # Python dependencies
├── .gitignore                            # Git ignore rules
└── README.md                             # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- USA Housing dataset from Kaggle

### Installation & Setup

1. **Navigate to the project directory**
   ```bash
   cd AIML_INTERN/TASK3
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

5. **Download the dataset**
   - Download the [USA Housing dataset](https://www.kaggle.com/datasets/vedavyasv/usa-housing) from Kaggle
   - Place the downloaded CSV file in the `data/` folder as `housing.csv`
   - Dataset should contain columns: `Avg. Area Income`, `Avg. Area House Age`, `Avg. Area Number of Rooms`, `Avg. Area Number of Bedrooms`, `Area Population`, `Price`, `Address`

6. **Run the script**
   ```bash
   python linear_regression.py
   ```

## 📊 Workflow

The script follows this workflow:

1. **Data Loading** - Load the USA Housing dataset from `data/housing.csv`
2. **Simple Linear Regression**
   - Use single feature: Area Population
   - Train-test split (80-20)
   - Train model and make predictions
   - Evaluate: MAE, MSE, RMSE, R²
   - Visualize: Scatter plot with regression line
3. **Multiple Linear Regression**
   - Use all numeric features (excluding Address)
   - Train-test split (80-20)
   - Train model and make predictions
   - Evaluate: MAE, MSE, RMSE, R²
   - Display feature coefficients
4. **Prediction** - Predict price for a sample house with custom features

## 📈 Expected Outputs

**Console Output:**
- Dataset preview (first 5 rows)
- Simple Linear Regression metrics (MAE, MSE, RMSE, R²)
- Multiple Linear Regression metrics (MAE, MSE, RMSE, R²)
- Feature coefficients table
- Sample house price prediction

**Generated Files:**
- `simple_linear_regression_plot.png` - Visualization of simple linear regression results

## 🔍 Key Concepts & Metrics

### Linear Regression Types:
- **Simple Linear Regression:** Uses one independent variable to predict the dependent variable
- **Multiple Linear Regression:** Uses multiple independent variables for prediction

### Evaluation Metrics:
- **MAE (Mean Absolute Error):** Average absolute difference between predictions and actual values
- **MSE (Mean Squared Error):** Average of squared differences between predictions and actual values
- **RMSE (Root Mean Squared Error):** Square root of MSE, in same units as target variable
- **R² (R-squared):** Proportion of variance in dependent variable predictable from independent variables (closer to 1 is better)

## 👤 Author

**Kumaran**
- 📧 Email: vvkumaran24@gmail.com
- 🎓 Project: AI/ML Internship - Linear Regression Task

---

## 📄 License

This project is open-source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

⭐ If you found this project helpful, please consider giving it a star!
