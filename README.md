# Machine Learning Pipelines for Classification, Regression, and NLP
This repository contains a comprehensive Jupyter notebook implementing three machine learning tasks with detailed pipelines. Below is a summary of the structure, methodologies, and results.

## Overview
### 1. **Classification**  
**Dataset**: `fars.csv` (vehicle accident injury severity prediction)  
**Pipelines**:  
1. **Decision Tree**:  
   - EDA, handling "Unknown" values, grouping rare categories, normalization.  
   - Feature encoding (one-hot), selection (Random Forest importance).  
   - SMOTE for class imbalance, GridSearchCV for hyperparameter tuning.  
   - Best model: **SVM (RBF kernel)** with 78.9% accuracy and 0.92 AUC-ROC.  

2. **SVM (RBF Kernel)**:  
   - Systematic sampling to reduce dataset size while preserving class distribution.  
   - Achieved highest accuracy and F1-score.  

3. **Random Forest**: Performed well but slightly lower than SVM.  
4. **KNN**: Struggled with minority classes.  

**Key Challenges**: Class imbalance, high dimensionality.  
**Best Model**: SVM with RBF kernel.

---

### 2. **Regression**  
**Dataset**: `fitting-results.csv` (predicting parameters `a`, `mu`, `tau`, `a0`)  
**Approach**:  
- Log-transformed skewed features (e.g., `n_cyanos`).  
- Standardized features, train-test-validation splits.  
- Tested four models per target:  
  - **Support Vector Regressor (SVR)**  
  - **Decision Tree**  
  - **Random Forest**  
  - **Polynomial Regression**  

**Results**:  
- **`a` and `mu`**: Random Forest outperformed others (test R² ≈ 0.98).  
- **`tau`**: All models performed well (R² ≈ 0.99).  
- **`a0`**: Challenging due to low variability; Random Forest achieved R² = 0.69.  

**Visualization**: Scatter plots of predicted vs. actual values for model comparison.  

---

### 3. **NLP**  
**Dataset**: `news.csv` (clustering news stories)  
**Pipelines**:  
1. **TF-IDF + K-Means**:  
   - Preprocessing: Lowercasing, stopword removal, lemmatization.  
   - Optimized `TfidfVectorizer` parameters via grid search.  
   - Silhouette Score: **0.021**.  

2. **Bag-of-Words (BoW) + K-Means**:  
   - Similar preprocessing, optimized `CountVectorizer`.  
   - Better clustering with Silhouette Score: **0.363**.  

**Key Terms**: Top cluster terms extracted (e.g., "game", "team" for sports news).  

---

## Dependencies  
- Python 3.10.4  
- Libraries:  
  - `scikit-learn`, `imbalanced-learn`, `pandas`, `numpy`, `seaborn`, `matplotlib`  
  - `nltk` (stopwords, lemmatization)  

## Usage  
1. Install dependencies:  
   ```bash
   pip install scikit-learn imbalanced-learn pandas numpy seaborn matplotlib nltk  
