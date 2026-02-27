Email Spam Classification 

This project is using different Statistical Machine Learning techniques to classify emails as Spam or Ham (Legitimate). 


Overview : 

The goal is to classify emails as Spam or Ham (Legitimate) using three distinct classification strategies and various text-representation techniques:

1. Naive Bayes: Uses a Bag-of-Words approach (CountVectorizer)

2. K-Nearest Neighbors (k-NN): Uses Dense Embeddings from a Pre-trained Transformer

3. Support Vector Machines (SVM): Uses Dense Embeddings and explores different kernels (Linear, Poly, RBF)

4. Dimensionality Reduction: Evaluates how PCA (Principal Component Analysis) affects performance and training efficiency

 Dataset
 
The dataset used is the  emails.csv .

Each email contains:

text → Email content ||  spam → Label (1 = Spam, 0 = Ham)

Dataset Split
The dataset is shuffled and divided into:

Training Set: Samples 1–2000

Validation Set: Samples 2001–3000

Test Set: Remaining samples

(Shuffling ensures both classes appear in all subsets)

 Models Implemented

1. Naive Bayes (Baseline)

A Multinomial Naive Bayes model using simple word counts. This serves as our baseline for speed and accuracy.

2. Sentence Embeddings (Transformer-based)

Make use of a Sentence-Transformer( paraphrase-multilingual-MiniLM-L12-v2 Sentence-Transformer ) to convert raw text into fixed-length 384-dimensional vectors. This captures semantic meaning that simple word counts miss.

3. Hyperparameter Tuning

The script automatically iterates through parameters to find the most "accurate" configuration:

k-NN: Tests k∈{1,3,5,11,15}.

SVM: Compares linear, poly, and rbf kernels.

4. Dimensionality Reduction (PCA)

To handle the high dimensionality of transformer embeddings,  apply PCA:

Variance Retention: We test keeping 90%, 95%, and 99% of the data's variance.

Extreme Compression: A Logistic Regression model trained on only 10 principal components.

Evaluation Metrics
 
  Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Validation data is used for model selection.
- Test data is used for final evaluation.

  How to Run

 Install Dependencies:
pip install pandas numpy nltk scikit-learn sentence-transformers

Run the Script:

Ensure emails.csv is in the root directory and run:


python main.py

