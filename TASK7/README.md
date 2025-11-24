TASK7 — SVM Classification

Structure:
- `data/` — place `breast_cancer.csv` here (download from Kaggle or let the script fallback to sklearn dataset)
- `src/svm_model.py` — training and plotting script for SVM (linear and RBF kernels)
- `visuals/` — output images: `linear_decision_boundary.png`, `rbf_decision_boundary.png`

Usage:
- From `TASK7/src/` run: `python svm_model.py`
- The script will try to load `../data/breast_cancer.csv`. If missing, it loads sklearn's dataset and saves a copy to `../data/breast_cancer.csv`.

Notes:
- The script selects the two features most correlated with the target to produce 2D decision boundary plots.
- Ensure dependencies are installed: `pandas, numpy, scikit-learn, matplotlib, seaborn`.
