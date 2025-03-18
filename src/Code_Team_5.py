# Import necessary libraries
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Function to display project information taking the team_members names as input
def my_information(team_member_1, team_member_2):
    print("=================== CSC790-IR Project ==============")
    print("Team 5")
    print("Team member 1: ", team_member_1)
    print("Team member 2 ", team_member_2)
    print("========================================================")

# Class for Naive Bayes classifier. 
class NaiveBayes:
    def _init_(self):
        #Initializes NaiveBayes class attributes.
        self.prior = {}
        self.condprob = {}
    
    #Trains the Naive Bayes classifier. 
    #The parameters are Training features and training labels respectively.
    def train(self, X_train, y_train):
        
        # Calculate prior probabilities
        self.prior = {c: sum(y_train == c) / len(y_train) for c in np.unique(y_train)}
        
        # Initialize conditional probabilities array
        self.condprob = {}
        
        # Convert sparse matrix to dense array
        X_train = X_train.toarray()
        
        # Calculate conditional probabilities
        for c in np.unique(y_train):
            X_train_c = X_train[y_train == c]
            total_count = X_train_c.sum(axis=0)
            total_count = total_count + 1  # Add smoothing
            total_count_sum = total_count.sum()
            cond_prob = total_count / (total_count_sum + len(X_train_c))
            self.condprob[c] = cond_prob


    # Predicts labels using the trained Naive Bayes classifier.
    # The parameters are Test Features and Predicted labels respectively
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            scores = {c: np.log(self.prior[c]) + np.sum(np.log(self.condprob[c][x.indices])) for c in self.prior}  
            predictions.append(max(scores, key=scores.get))
        return predictions

# Function to evaluate model based on the inputs i.e.., True labels and Pedicted labels
# The Evaluation metrics are Accuracy, Precision, Recall and F1-score.
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# This function performs evaluation for each set of features.

#This function first takes the 'features' and 'labels' as input and 
#then applies above Naive Bayes classifictaion and predicts the labels.
#Finally based on the predicted and true labels, evaluation is 
# performed using the above evaluate_model function and returns the accuracy, precision, recall, f1.
def evaluate_features(features, labels):
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Train Naive Bayes classifier
    nb_classifier = NaiveBayes()
    nb_classifier.train(X_train, y_train)
    
    # Predict labels
    y_pred = nb_classifier.predict(X_test)
    
    # Evaluate model
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
    
    return accuracy, precision, recall, f1

# This function is used to load stopwords from a file
def load_stopwords(filepath):
    with open(filepath, "r") as file:
        stopwords_list = file.read().splitlines()
    return stopwords_list

# This function is used to load special characters from a file
def load_special_chars(filepath):
    with open(filepath, "r") as file:
        special_chars = file.read().strip()
    return special_chars

#This function is mainly used for preprocessing
#This function preprocesses text by removing HTML tags, special characters,
# non-ASCII characters, numbers, stopwords, and performing stemming and 
#then finally returns preprocessed tokens.
def preprocess_text(review, stopwords_list, special_chars):
    
    # Remove HTML tags if any
    review = re.sub(r'<[^>]+>', '', review)
    
    # Remove special characters
    review = ''.join(char for char in review if char not in special_chars)

    # Remove non-ASCII characters
    review = ''.join(char for char in review if ord(char) < 128)

    # Remove numbers
    review = re.sub(r'\d+', '', review)
    
    # Tokenize review
    tokens = word_tokenize(review)
    
    # Convert words to lowercase
    tokens = [word.lower() for word in tokens]
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords_list]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return tokens

# Display project information
my_information("Mounika Gullapalli","Leela Chittoori")

# Load stopwords and special characters using above declared 
#functions to load stopwords file and special characters file
stopwords_list = load_stopwords("stopwords.txt")
special_chars = load_special_chars("special-chars.txt")

# Preprocess each review in the CSV file
#Uses the above preprocess_text function to preprocess each review
preprocessed_documents = []
with open("IMDB.csv", "r", encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skips header
    for row in csv_reader:
        review = row[0]  # Assuming the review is in the first column
        label = 1 if row[1] == "positive" else 0  # Assuming the label is in the second column
        preprocessed_review = preprocess_text(review, stopwords_list, special_chars)
        preprocessed_documents.append((preprocessed_review, label))

# Convert preprocessed documents into strings
preprocessed_reviews_strings = [' '.join(review) for review, _ in preprocessed_documents]

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_reviews_strings)

# Extract labels from preprocessed documents
labels = [label for _, label in preprocessed_documents]

# Initialize lists to store F1 scores for each feature selection method
f1_scores_mi = []
f1_scores_chi2 = []
f1_scores_freq = []

# Initialize lists to store all evaluation metrics
evaluation_metrics_mi = []
evaluation_metrics_chi2 = []
evaluation_metrics_freq = []

# List of k values for ploting K vs F1 Score graph
k_values = [50,100,150,200,250]

# Loop over k values
for k in k_values:
    # Perform mutual information feature selection
    mutual_info_selector = SelectKBest(score_func=mutual_info_classif, k=k)
    mutual_info_selector.fit(tfidf_matrix, [label for _, label in preprocessed_documents])
    mutual_info_indices = mutual_info_selector.get_support(indices=True)
    
    # Perform chi-square feature selection
    chi_square_selector = SelectKBest(score_func=chi2, k=k)
    chi_square_selector.fit(tfidf_matrix, [label for _, label in preprocessed_documents])
    chi_square_indices = chi_square_selector.get_support(indices=True)
    
    # Perform frequency-based feature selection
    term_counts = Counter()
    for review in [review for review, _ in preprocessed_documents]:
        term_counts.update(review)
    frequency_indices = [vectorizer.vocabulary_[term] for term, _ in term_counts.most_common(k)]
    
    # Evaluate each set of selected features
    accuracy_mi, precision_mi, recall_mi, f1_mi = evaluate_features(tfidf_matrix[:, mutual_info_indices], labels)
    accuracy_chi2, precision_chi2, recall_chi2, f1_chi2 = evaluate_features(tfidf_matrix[:, chi_square_indices], labels)
    accuracy_freq, precision_freq, recall_freq, f1_freq = evaluate_features(tfidf_matrix[:, frequency_indices], labels)
    
    # Print evaluation metrics for each feature selection method
    print(f"\nEvaluation Metrics for k={k}:")
    print("Mutual Information Feature Selection:")
    print(f"Accuracy: {accuracy_mi}, Precision: {precision_mi}, Recall: {recall_mi}, F1 Score: {f1_mi}")
    
    print("\nChi-Square Feature Selection:")
    print(f"Accuracy: {accuracy_chi2}, Precision: {precision_chi2}, Recall: {recall_chi2}, F1 Score: {f1_chi2}")
    
    print("\nFrequency-Based Feature Selection:")
    print(f"Accuracy: {accuracy_freq}, Precision: {precision_freq}, Recall: {recall_freq}, F1 Score: {f1_freq}")

    # Store F1 scores for each k value
    f1_scores_mi.append(f1_mi)
    f1_scores_chi2.append(f1_chi2)
    f1_scores_freq.append(f1_freq)
    
    # Store all evaluation metrics for each k value and feature selection method
    evaluation_metrics_mi.append((k, f1_mi))
    evaluation_metrics_chi2.append((k, f1_chi2))
    evaluation_metrics_freq.append((k, f1_freq))

# Print evaluation metrics for each feature selection method
print("Mutual Information Feature Selection:")
for k, f1_mi in evaluation_metrics_mi:
    print(f"k={k}: F1 Score={f1_mi}")

print("\nChi-Square Feature Selection:")
for k, f1_chi2 in evaluation_metrics_chi2:
    print(f"k={k}: F1 Score={f1_chi2}")

print("\nFrequency-Based Feature Selection:")
for k, f1_freq in evaluation_metrics_freq:
    print(f"k={k}: F1 Score={f1_freq}")

# Plot the results i.e., graph for K vs F1 Score.
plt.plot(k_values, f1_scores_mi, label='Mutual Information')
plt.plot(k_values, f1_scores_chi2, label='Chi-Square')
plt.plot(k_values, f1_scores_freq, label='Frequency-Based')
plt.xlabel('Number of Features (k)')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Number of Features for Different Feature Selection Methods')
plt.legend()
plt.show()


#We have used inbuilt functions for feature selection but have 
#coded the Naive Bayes function from scratch using the lecture notes.

#This code finally displays accuracy, precision, recall, 
#F1 for all the three feature selection methods at all the mentioned k values in 'k_values'

#Also, prints all the k values and respective F1 score for each feature selection method 
#This is to facilitate easy generation of graph 

#Finally, the graph : 'Number of features vs F1' is plotted.

#From the graph, we have found that features selected by Chi-Square method does great job in 
# classifying the reviews more accurately compared to features selected from the other 2 feature selection methods.

#We have developed this code using Visual Studio
#Command to run the code: python Team_5.py

#This code for analysing the top 10 functions taking more time is present in 'Time_Analysis_code_Team_5.py'
#We have submitted it as a seperate file for easier understanding.