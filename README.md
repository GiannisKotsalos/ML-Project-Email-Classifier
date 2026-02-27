#  Email Spam Classification

## Machine Learning & Transformer-Based Text Classification

This project builds and compares multiple **Statistical Machine Learning models** to classify emails as **Spam** or **Ham (Legitimate)**.

It combines **traditional NLP techniques** with **modern transformer-based embeddings**, while evaluating model performance, computational efficiency, and dimensionality trade-offs.

---

## Project Overview

The objective is to design and evaluate different classification pipelines to answer:

- How well do classical NLP methods perform compared to transformer embeddings?
- Does semantic representation improve spam detection accuracy?
- How do different SVM kernels affect classification performance?
- Can dimensionality reduction maintain accuracy while improving efficiency?

---

##  Dataset

**File:** `emails.csv`

Each email contains:

- `text` → Email content  
- `spam` → Label (1 = Spam, 0 = Ham)

###  Data Split Strategy

- **Training Set:** Samples 1–2000  
- **Validation Set:** Samples 2001–3000  
- **Test Set:** Remaining samples  

The dataset is shuffled before splitting to ensure balanced class representation across all subsets.

- Validation set → Model selection  
- Test set → Final performance evaluation  

---

##  Models Implemented

### 1 Multinomial Naive Bayes (Baseline)
- Representation: **Bag-of-Words (CountVectorizer)**
- Fast and interpretable classical ML baseline

### 2 k-Nearest Neighbors (k-NN)
- Input: Transformer-based dense embeddings  
- Hyperparameter tuning:
- k ∈ {1, 3, 5, 11, 15}
  
### 3 Support Vector Machines (SVM)
- Input: Transformer embeddings  
- Kernels compared:
- Linear
- Polynomial
- RBF

### 4 Dimensionality Reduction (PCA)
- Variance retention experiments:
- 90%
- 95%
- 99%
- Extreme compression test:
- Logistic Regression using 10 principal components

---

##  Evaluation Metrics

Models are evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  

Validation data is used for model selection.  
Test data is used for final evaluation.

---

##  Tech Stack

- Python  
- Pandas  
- NumPy  
- NLTK  
- scikit-learn  
- sentence-transformers  

---

##  How to Run

Ensure emails.csv is in the root directory:

### Install Dependencies

```bash
pip install pandas numpy nltk scikit-learn sentence-transformers


python main.py

