import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import re
import nltk

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer

df = pd.read_csv('emails.csv')

# Cleaning the Text from unessesery characters
stop_words = set(stopwords.words('english')) 

def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase and split into words
    words = text.lower().split() 
    # Remove stop words
    cleaned = [w for w in words if w not in stop_words]
    return " ".join(cleaned) 

df['text'] = df['text'].apply(clean_text)

# Shuffle the dataset to ensure balanced distribution across splits
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the dataset
# Samples 1-2000 (0-1999 index), 2001-3000 (2000-2999 index), and the rest
train_df = df.iloc[0:2000]
val_df = df.iloc[2000:3000]
test_df = df.iloc[3000:]

def NaiveBayesClassifier(train_df, val_df, test_df):
  
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()
    
    # Fit and transform the training data
    train_vectors = vectorizer.fit_transform(train_df['text'])
    
    # Transform the validation and test data
    val_vectors = vectorizer.transform(val_df['text'])
    test_vectors = vectorizer.transform(test_df['text'])
    
    # Initialize the MultinomialNB classifier   
    classifier = MultinomialNB()
    
    # Fit the classifier on the training data
    classifier.fit(train_vectors, train_df['spam'])
    
    # Predict the labels for the validation and test data
    val_predictions = classifier.predict(val_vectors)
    test_predictions = classifier.predict(test_vectors)
    
    # Get probability predictions for AUC calculation
    val_proba = classifier.predict_proba(val_vectors)[:, 1]  # Probability of class 1 (spam)
    test_proba = classifier.predict_proba(test_vectors)[:, 1]  # Probability of class 1 (spam)
    
    # Calculate AUC scores
    val_auc = roc_auc_score(val_df['spam'], val_proba)
    test_auc = roc_auc_score(test_df['spam'], test_proba)
    
    # Print the classification report
    print("Validation Set Classification Report:")
    print(classification_report(val_df['spam'], val_predictions, zero_division=0))
    print(f"Validation Set AUC: {val_auc:.4f}\n")
    
    print("Test Set Classification Report:")
    print(classification_report(test_df['spam'], test_predictions, zero_division=0))
    print(f"Test Set AUC: {test_auc:.4f}")
    
    # Return metrics (C-style: functions return values)
    return {
        'val_auc': val_auc,
        'test_auc': test_auc,
        'val_predictions': val_predictions,
        'test_predictions': test_predictions
    }


# Call Naive Bayes function
NaiveBayesClassifier(train_df, val_df, test_df)

def kNNClassifier(train_df, val_df, test_df, X_train_embed, X_val_embed, X_test_embed):
    """
    K-Nearest Neighbors Classifier function
    Finds the best k value and evaluates on validation and test sets
    
    Parameters:
    train_df: Training dataset
    val_df: Validation dataset
    test_df: Test dataset
    X_train_embed: Training embeddings
    X_val_embed: Validation embeddings
    X_test_embed: Test embeddings
    
    Returns:
    Dictionary containing metrics and best k value
    """
    # Variables to track the best performance
    best_k = 0
    best_knn_score = 0
    
    print("--- Step 4: Finding the Best k for K-NN ---")
    
    # We loop through several k values to find the best one as requested
    for k in [1, 3, 5, 11, 15]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_embed, train_df['spam'])
        
        # Validation accuracy to determine the "best" k
        val_accuracy = knn.score(X_val_embed, val_df['spam'])
        print(f"Testing k={k}: Validation Accuracy = {val_accuracy:.4f}")
        
        # Track the best result
        if val_accuracy > best_knn_score:
            best_knn_score = val_accuracy
            best_k = k
    
    # Final Evaluation of the BEST model
    print(f"\nBest result found: k={best_k} with {best_knn_score:.4f} accuracy")
    
    # Re-run or use the best model to get AUC for the report
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train_embed, train_df['spam'])
    
    # Get predictions
    val_predictions = best_knn.predict(X_val_embed)
    test_predictions = best_knn.predict(X_test_embed)
    
    # Use probabilities for AUC as per requirement #2
    val_probs = best_knn.predict_proba(X_val_embed)[:, 1]
    test_probs = best_knn.predict_proba(X_test_embed)[:, 1]
    
    val_auc = roc_auc_score(val_df['spam'], val_probs)
    test_auc = roc_auc_score(test_df['spam'], test_probs)
    
    print(f"Final Validation AUC (k={best_k}): {val_auc:.4f}")
    print(f"Final Test AUC (k={best_k}): {test_auc:.4f}")
    
    # Return metrics (C-style: functions return values)
    return {
        'val_auc': val_auc,
        'test_auc': test_auc,
        'val_predictions': val_predictions,
        'test_predictions': test_predictions,
        'best_k': best_k,
        'best_knn_score': best_knn_score
    }

def SVMClassifier(train_df, val_df, test_df, X_train_embed, X_val_embed, X_test_embed):
    """
    Support Vector Machine Classifier function
    Finds the best kernel and evaluates on validation and test sets
    
    Parameters:
    train_df: Training dataset
    val_df: Validation dataset
    test_df: Test dataset
    X_train_embed: Training embeddings
    X_val_embed: Validation embeddings
    X_test_embed: Test embeddings
    
    Returns:
    Dictionary containing metrics and best kernel
    """
    kernels = ['linear', 'poly', 'rbf']
    best_svm_accuracy = 0
    best_kernel_name = ""
    
    print("--- Step 5: Finding the Best Kernel for SVM ---")
    
    for k_type in kernels:
        svm = SVC(kernel=k_type)
        svm.fit(X_train_embed, train_df['spam'])
        
        val_accuracy = svm.score(X_val_embed, val_df['spam'])
        print(f"Testing kernel={k_type}: Validation Accuracy = {val_accuracy:.4f}")
        
        if val_accuracy > best_svm_accuracy:
            best_svm_accuracy = val_accuracy
            best_kernel_name = k_type
    
    print(f"Best result: kernel={best_kernel_name} with {best_svm_accuracy:.4f} accuracy")
    
    # Create final model with probability=True for AUC calculation
    best_svm = SVC(kernel=best_kernel_name, probability=True)
    best_svm.fit(X_train_embed, train_df['spam'])
    
    # Get predictions
    val_predictions = best_svm.predict(X_val_embed)
    test_predictions = best_svm.predict(X_test_embed)
    
    # Use probabilities for AUC as per requirement
    val_probs = best_svm.predict_proba(X_val_embed)[:, 1]
    test_probs = best_svm.predict_proba(X_test_embed)[:, 1]
    
    val_auc = roc_auc_score(val_df['spam'], val_probs)
    test_auc = roc_auc_score(test_df['spam'], test_probs)
    
    print(f"Final Validation AUC (kernel={best_kernel_name}): {val_auc:.4f}")
    print(f"Final Test AUC (kernel={best_kernel_name}): {test_auc:.4f}")
    
    # Return metrics (C-style: functions return values)
    return {
        'val_auc': val_auc,
        'test_auc': test_auc,
        'val_predictions': val_predictions,
        'test_predictions': test_predictions,
        'best_kernel': best_kernel_name,
        'best_svm_accuracy': best_svm_accuracy
    }

# Initialize the sentence transformer model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# Calculate embeddings once globally
print("\nEncoding emails into embeddings...")
X_train_embed = model.encode(train_df['text'].tolist())
X_val_embed = model.encode(val_df['text'].tolist())
X_test_embed = model.encode(test_df['text'].tolist())
print(f"Training embeddings shape: {X_train_embed.shape}")
print(f"Validation embeddings shape: {X_val_embed.shape}")
print(f"Test embeddings shape: {X_test_embed.shape}")

# Then pass these pre-calculated arrays to your functions
kNN_results = kNNClassifier(train_df, val_df, test_df, X_train_embed, X_val_embed, X_test_embed)
SVM_results = SVMClassifier(train_df, val_df, test_df, X_train_embed, X_val_embed, X_test_embed)

from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression

def PCAmethod(X_train_embed, X_val_embed, train_labels, val_labels, best_kernel):
   
    variances = [0.90, 0.95, 0.99]
    results = {}

    print(f"\n--- Βήμα 6: PCA με SVM (Kernel: {best_kernel}) ---")
    
    for v in variances:
        # Εφαρμογή PCA 
        pca = PCA(n_components=v)
        X_train_pca = pca.fit_transform(X_train_embed)
        X_val_pca = pca.transform(X_val_embed)
        
        # Εκπαίδευση SVM στα μειωμένα δεδομένα 
        svm_pca = SVC(kernel=best_kernel)
        svm_pca.fit(X_train_pca, train_labels)
        
        accuracy = svm_pca.score(X_val_pca, val_labels)
        num_components = pca.n_components_
        
        print(f"Μεταβλητότητα {v*100}%: Διαστάσεις={num_components}, Accuracy={accuracy:.4f}")
        results[v] = {'accuracy': accuracy, 'dims': num_components}
    
    return results # Έξω από το loop για να ολοκληρωθούν όλες οι δοκιμές 

def LogisticRegressionMethod(X_train_embed, X_val_embed, train_labels, val_labels):
    
    print("\n--- Βήμα 7: PCA (10 διαστάσεις) & Λογιστική Παλινδρόμηση ---")
    
    # Μείωση σε 10 διαστάσεις 
    pca10 = PCA(n_components=10)
    X_train_10 = pca10.fit_transform(X_train_embed)
    X_val_10 = pca10.transform(X_val_embed)
    
    # Λογιστική Παλινδρόμηση 
    lr = LogisticRegression()
    lr.fit(X_train_10, train_labels)
    lr_acc = lr.score(X_val_10, val_labels)
    
    print(f"Accuracy με 10 διαστάσεις: {lr_acc:.4f}")
    return lr_acc

# Εκτέλεση των τελικών βημάτων
PCA_results = PCAmethod(X_train_embed, X_val_embed, train_df['spam'], val_df['spam'], SVM_results['best_kernel'])
LR_results = LogisticRegressionMethod(X_train_embed, X_val_embed, train_df['spam'], val_df['spam'])