# 📊 Student Performance Exploratory Data Analysis (EDA)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Latest-green.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-Latest-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📝 Description

A comprehensive Exploratory Data Analysis (EDA) on the cleaned **Student Performance Dataset**. This project analyzes the relationships between various factors (gender, lunch type, test preparation) and student scores in math, reading, and writing. It includes statistical summaries, visualizations, and insights to understand the data distribution and correlations.

## 🎯 Objective

The main objective of this project is to perform EDA on the cleaned Student Performance dataset by:
- **Analyzing data distribution** using histograms and boxplots
- **Exploring correlations** between numerical variables
- **Investigating categorical impacts** on scores (gender, lunch, test prep)
- **Generating visualizations** for better understanding
- **Providing automatic insights** based on statistical analysis

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

## 📁 Project Structure

```
TASK2/
│
├── cleaned_students_data.csv        # Input: Cleaned dataset from TASK1
├── EDA.py                           # Main EDA script
├── .gitignore                       # Git ignore rules
├── README.md                        # Project documentation
├── HISTOGRAM.png                    # Histogram visualization (generated)
├── BOXPLOT.png                      # Boxplot visualization (generated)
├── HEATMAP.png                      # Correlation heatmap (generated)
├── PAIRPLOT.png                     # Pairplot visualization (generated)
├── CAT1.png                         # Gender vs Math score (generated)
├── CAT2.png                         # Lunch vs Math score (generated)
└── CAT3.png                         # Test prep vs Writing score (generated)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Cleaned dataset from TASK1 (`cleaned_students_data.csv`)

### Installation & Setup

1. **Navigate to the project directory**
   ```bash
   cd AIML_INTERN/TASK2
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
   pip install pandas numpy matplotlib seaborn
   ```

5. **Run the EDA script**
   ```bash
   python EDA.py
   ```

## 📊 Expected Output

After running the script, you will receive:

1. **Console Output:**
   - Dataset overview (shape, columns, data types)
   - Null values and duplicate check
   - Summary statistics for numeric features
   - Value counts for categorical variables
   - Automatic insights (mean scores, correlations, group comparisons)

2. **Generated Visualizations:**
   - **HISTOGRAM.png** - Distribution of math, reading, and writing scores
   - **BOXPLOT.png** - Boxplots showing score distributions and outliers
   - **HEATMAP.png** - Correlation matrix between all numeric variables
   - **PAIRPLOT.png** - Pairwise relationships between scores
   - **CAT1.png** - Math score vs Gender analysis
   - **CAT2.png** - Math score vs Lunch type analysis
   - **CAT3.png** - Writing score vs Test preparation analysis

## 📈 EDA Process

The analysis follows these steps:

1. **Data Loading** - Import the cleaned Student Performance dataset
2. **Dataset Overview** - Display basic information and check for issues
3. **Statistical Summary** - Generate descriptive statistics
4. **Univariate Analysis** - Histograms and boxplots for score distributions
5. **Bivariate Analysis** - Correlation matrix and pairplots
6. **Categorical Analysis** - Impact of categorical variables on scores
7. **Automatic Insights** - Statistical summaries and key findings

## 🔍 Key Insights

The EDA provides insights such as:
- Distribution patterns of student scores
- Correlations between different subjects
- Impact of gender on academic performance
- Effect of lunch type on scores
- Influence of test preparation on writing scores
- Statistical comparisons between different groups

## 👤 Author

**Kumaran**
- 📧 Email: vvkumaran24@gmail.com
- 🎓 Project: AI/ML Internship - EDA Task

---

## 📄 License

This project is open-source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

⭐ If you found this project helpful, please consider giving it a star!
