import os
    # Silence the tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import re
import nltk

# Make sure to have the stopwords downloaded (only if not already downloaded to avoid errors    )
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

# get rid of all the unnecessary stuff in the text to avoid errors
# define the stop words to avoid errors
stop_words = set(stopwords.words('english')) 

def clean_text(text):
    
    # Strip out all the special characters and numbers - only want letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Make everything lowercase and break it into individual words
    words = text.lower().split() 
    # Filter out those common stop words that don't help us
    cleaned = [w for w in words if w not in stop_words]
    return " ".join(cleaned) 

df['text'] = df['text'].apply(clean_text)

# Shuffle everything up first to get a good mix in each set
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# First 2000 go to training, next 1000 to validation, rest to testing
train_df = df.iloc[0:2000]
val_df = df.iloc[2000:3000]
test_df = df.iloc[3000:]

def NaiveBayesClassifier(train_df, val_df, test_df):
  
    # Set up our vectorizer to turn text into numbers to convert text into vectors
    vectorizer = CountVectorizer()
    
    # Learn from training data and convert it to vectors
    train_vectors = vectorizer.fit_transform(train_df['text'])
    
    # Convert validation and test data using what we learned (don't refit!)
    val_vectors = vectorizer.transform(val_df['text'])
    test_vectors = vectorizer.transform(test_df['text'])
    
    # Create our Naive Bayes classifier
    classifier = MultinomialNB()
    
    # Train the classifier using the training data
    classifier.fit(train_vectors, train_df['spam'])
    
        # See what it thinks about validation and test data to get predictions
    val_predictions = classifier.predict(val_vectors)
    test_predictions = classifier.predict(test_vectors)
    
    # Get the probabilities too -  need  for AUC
    val_proba = classifier.predict_proba(val_vectors)[:, 1]  # This is the spam probability
    test_proba = classifier.predict_proba(test_vectors)[:, 1]  # Same here
    
    # Calculate how well we did with AUC scores
    val_auc = roc_auc_score(val_df['spam'], val_proba)
    test_auc = roc_auc_score(test_df['spam'], test_proba)
    
    # Show the results
    print("Validation Set Classification Report:")
    print(classification_report(val_df['spam'], val_predictions, zero_division=0))
    print(f"Validation Set AUC: {val_auc:.4f}\n")
    
    print("Test Set Classification Report:")
    print(classification_report(test_df['spam'], test_predictions, zero_division=0))
    print(f"Test Set AUC: {test_auc:.4f}")
    
    # Return everything  to use it later 
    return {
        'val_auc': val_auc,
        'test_auc': test_auc,
        'val_predictions': val_predictions,
        'test_predictions': test_predictions
    }


#  run Naive Bayes and find the best k value
NaiveBayesClassifier(train_df, val_df, test_df)

def kNNClassifier(train_df, val_df, test_df, X_train_embed, X_val_embed, X_test_embed):
   
    # Keep track of which k value works best
    best_k = 0
    best_knn_score = 0
    
    print("---  Finding the Best k for K-NN ---")
    
    # Try out different k values and see which one gives us the best results
    for k in [1, 3, 5, 11, 15]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_embed, train_df['spam'])
        
        # Check how well this k does on validation data
        val_accuracy = knn.score(X_val_embed, val_df['spam'])
        print(f"Testing k={k}: Validation Accuracy = {val_accuracy:.4f}")
        
        # If this is better than what we've seen, remember it
        if val_accuracy > best_knn_score:
            best_knn_score = val_accuracy
            best_k = k
    
    # Found  best k it is
    print(f"\nBest result found: k={best_k} with {best_knn_score:.4f} accuracy")
    
    # Now train the model again with the best k  to get proper predictions
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train_embed, train_df['spam'])
    
    # Make predictions on both validation and test sets
    val_predictions = best_knn.predict(X_val_embed)
    test_predictions = best_knn.predict(X_test_embed)
    
    # Get probabilities too -  need  for AUC
    val_probs = best_knn.predict_proba(X_val_embed)[:, 1]
    test_probs = best_knn.predict_proba(X_test_embed)[:, 1]
    
    val_auc = roc_auc_score(val_df['spam'], val_probs)
    test_auc = roc_auc_score(test_df['spam'], test_probs)
    
    print(f"Final Validation AUC (k={best_k}): {val_auc:.4f}")
    print(f"Final Test AUC (k={best_k}): {test_auc:.4f}")
    
    # Return all the good stuff
    return {
        'val_auc': val_auc,
        'test_auc': test_auc,
        'val_predictions': val_predictions,
        'test_predictions': test_predictions,
        'best_k': best_k,
        'best_knn_score': best_knn_score
    }

def SVMClassifier(train_df, val_df, test_df, X_train_embed, X_val_embed, X_test_embed):
   
    kernels = ['linear', 'poly', 'rbf']
    best_svm_accuracy = 0
    best_kernel_name = ""
    
    print("--- Finding the Best Kernel for SVM ---")
    
    # Try each kernel type and see which one works best
    for k_type in kernels:
        svm = SVC(kernel=k_type)
        svm.fit(X_train_embed, train_df['spam'])
        
        val_accuracy = svm.score(X_val_embed, val_df['spam'])
        print(f"Testing kernel={k_type}: Validation Accuracy = {val_accuracy:.4f}")
        
        # If this kernel is better save it
        if val_accuracy > best_svm_accuracy:
            best_svm_accuracy = val_accuracy
            best_kernel_name = k_type
    
    print(f"Best result: kernel={best_kernel_name} with {best_svm_accuracy:.4f} accuracy")
    
    # Build the final model with probability enabled so we can calculate AUC
    best_svm = SVC(kernel=best_kernel_name, probability=True)
    best_svm.fit(X_train_embed, train_df['spam'])
    
    # Make predictions on both sets
    val_predictions = best_svm.predict(X_val_embed)
    test_predictions = best_svm.predict(X_test_embed)
    
    # Get probabilities for AUC calculation
    val_probs = best_svm.predict_proba(X_val_embed)[:, 1]
    test_probs = best_svm.predict_proba(X_test_embed)[:, 1]
    
    val_auc = roc_auc_score(val_df['spam'], val_probs)
    test_auc = roc_auc_score(test_df['spam'], test_probs)
    
    print(f"Final Validation AUC (kernel={best_kernel_name}): {val_auc:.4f}")
    print(f"Final Test AUC (kernel={best_kernel_name}): {test_auc:.4f}")
    
    # Return everything we found
    return {
        'val_auc': val_auc,
        'test_auc': test_auc,
        'val_predictions': val_predictions,
        'test_predictions': test_predictions,
        'best_kernel': best_kernel_name,
        'best_svm_accuracy': best_svm_accuracy
    }

# Load up the sentence transformer model to convert text into vectors
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# Convert all our emails into embeddings - do this once  don't  repeat 
print("\nEncoding emails into embeddings...")
X_train_embed = model.encode(train_df['text'].tolist())
X_val_embed = model.encode(val_df['text'].tolist())
X_test_embed = model.encode(test_df['text'].tolist())
print(f"Training embeddings shape: {X_train_embed.shape}")
print(f"Validation embeddings shape: {X_val_embed.shape}")
print(f"Test embeddings shape: {X_test_embed.shape}")

# use these embeddings for kNN and SVM
kNN_results = kNNClassifier(train_df, val_df, test_df, X_train_embed, X_val_embed, X_test_embed)
SVM_results = SVMClassifier(train_df, val_df, test_df, X_train_embed, X_val_embed, X_test_embed)

from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression

def PCAmethod(X_train_embed, X_val_embed, train_labels, val_labels, best_kernel):
   
    variances = [0.90, 0.95, 0.99]
    results = {}

    print(f"\n---  PCA with SVM (Kernel: {best_kernel}) ---")
    
    for v in variances:
        # Apply PCA to reduce dimensions while keeping this much variance
        pca = PCA(n_components=v)
        X_train_pca = pca.fit_transform(X_train_embed)
        X_val_pca = pca.transform(X_val_embed)
        
        # Train SVM on the reduced data
        svm_pca = SVC(kernel=best_kernel)
        svm_pca.fit(X_train_pca, train_labels)
        
        accuracy = svm_pca.score(X_val_pca, val_labels)
        num_components = pca.n_components_
        
        print(f"Variance {v*100}%: Dimensions={num_components}, Accuracy={accuracy:.4f}")
        results[v] = {'accuracy': accuracy, 'dims': num_components}
    
    return results # Return after trying all variance levels 

def LogisticRegressionMethod(X_train_embed, X_val_embed, train_labels, val_labels):
    
    print("\n--- PCA (10 dimensions) & Logistic Regression ---")
    
    # Reduce everything down to just 10 dimensions
    pca10 = PCA(n_components=10)
    X_train_10 = pca10.fit_transform(X_train_embed)
    X_val_10 = pca10.transform(X_val_embed)
    
    # Train logistic regression on the reduced data
    lr = LogisticRegression()
    lr.fit(X_train_10, train_labels)
    lr_acc = lr.score(X_val_10, val_labels)
    
    print(f"Accuracy with 10 dimensions: {lr_acc:.4f}")
    return lr_acc

# Run the final methods - PCA with SVM and Logistic Regression
PCA_results = PCAmethod(X_train_embed, X_val_embed, train_df['spam'], val_df['spam'], SVM_results['best_kernel'])
LR_results = LogisticRegressionMethod(X_train_embed, X_val_embed, train_df['spam'], val_df['spam'])