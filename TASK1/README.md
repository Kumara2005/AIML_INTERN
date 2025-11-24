# 📊 Student Performance Data Cleaning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Latest-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📝 Description

A comprehensive data cleaning and preprocessing pipeline for the **Student Performance Dataset**. This project systematically handles missing values, removes duplicates, detects and treats outliers, and prepares the dataset for further analysis or machine learning applications.

## 🎯 Objective

The main objective of this project is to clean and preprocess the Student Performance dataset by:
- **Removing null values** and handling missing data appropriately
- **Eliminating duplicate records** to ensure data integrity
- **Detecting and treating outliers** using statistical methods (IQR)
- **Generating a clean dataset** ready for analysis and modeling
- **Providing detailed summary statistics** before and after cleaning

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Feature scaling and preprocessing

## 📁 Project Structure

```
TASK1/
│
├── StudentsPerformance.csv          # Original dataset
├── datacleaning.py                  # Main data cleaning script
├── cleaned_students_data.csv        # Cleaned and processed dataset
├── .gitignore                       # Git ignore rules
└── README.md                        # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AIML_INTERN/TASK1
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
   pip install pandas numpy scikit-learn
   ```

5. **Run the data cleaning script**
   ```bash
   python datacleaning.py
   ```

## 📊 Expected Output

After running the script, you will receive:

1. **Console Output:**
   - Initial dataset shape and information
   - Count of missing values before and after handling
   - Number of duplicate records detected and removed
   - Outlier detection results for numerical columns
   - Final dataset shape and summary statistics

2. **Generated File:**
   - `cleaned_students_data.csv` - The cleaned dataset with:
     - No missing values
     - No duplicate records
     - Outliers handled appropriately
     - All data types validated

## 📈 Data Cleaning Process

The cleaning pipeline follows these steps:

1. **Data Loading** - Import the raw Student Performance dataset
2. **Initial Inspection** - Display dataset info, shape, and summary statistics
3. **Missing Value Analysis** - Identify and handle null values (fill with mode for categorical, mean for numerical)
4. **Categorical Encoding** - Convert categorical variables to binary features using one-hot encoding
5. **Outlier Detection & Removal** - Use IQR method to identify and remove outliers in score columns
6. **Feature Scaling** - Apply standardization to numerical score columns using StandardScaler
7. **Data Export** - Save the cleaned and processed dataset to CSV format

## 👤 Author

**Kumaran**
- 📧 Email: vvkumaran24@gmail.com
- 🎓 Project: AI/ML Internship - Data Cleaning Task

---

## 📄 License

This project is open-source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

⭐ If you found this project helpful, please consider giving it a star!
