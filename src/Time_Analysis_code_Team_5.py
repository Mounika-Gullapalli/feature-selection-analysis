#This code is mainly for time analysis

#For the time analysis part, we need to paste the code in the main function
#and run it using 'cProfile’ and ’pstats’ packages.

#import necessary packages for time analysis
import cProfile 
import pstats

# Define your main function
def main():
    # code is pasted here
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

    # Modify the NaiveBayes class
    class NaiveBayes:
        def init(self):
            self.prior = {}
            self.condprob = {}
        
            # Modify the train method in the NaiveBayes class
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


        # Modify the predict method in the NaiveBayes class
        def predict(self, X_test):
            predictions = []
            for x in X_test:
                scores = {c: np.log(self.prior[c]) + np.sum(np.log(self.condprob[c][x.indices])) for c in self.prior}  # Modify this line
                predictions.append(max(scores, key=scores.get))
            return predictions



    # Define function to evaluate model
    def evaluate_model(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1

    # Define function to perform evaluation for each set of features
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

    def load_stopwords(filepath):
        with open(filepath, "r") as file:
            stopwords_list = file.read().splitlines()
        return stopwords_list

    def load_special_chars(filepath):
        with open(filepath, "r") as file:
            special_chars = file.read().strip()
        return special_chars

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

    # Load stopwords and special characters
    stopwords_list = load_stopwords("stopwords.txt")
    special_chars = load_special_chars("special-chars.txt")

    # Preprocess each review in the CSV file
    preprocessed_documents = []
    with open("IMDB.csv", "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header
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

    # List of k values
    k_values = [100]

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
        _, _, _, f1_mi = evaluate_features(tfidf_matrix[:, mutual_info_indices], labels)
        _, _, _, f1_chi2 = evaluate_features(tfidf_matrix[:, chi_square_indices], labels)
        _, _, _, f1_freq = evaluate_features(tfidf_matrix[:, frequency_indices], labels)
        
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

    # Plot the results
    plt.plot(k_values, f1_scores_mi, label='Mutual Information')
    plt.plot(k_values, f1_scores_chi2, label='Chi-Square')
    plt.plot(k_values, f1_scores_freq, label='Frequency-Based')
    plt.xlabel('Number of Features (k)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Number of Features for Different Feature Selection Methods')
    plt.legend()
    plt.show()

# Run the main function under the profiler
cProfile.run('main()', 'profile_stats')

# Analyze the profiling results
# It will run the code and gives the top 
#10 functions that are taking more time to execute.
import pstats
p = pstats.Stats('profile_stats')
p.sort_stats('cumulative').print_stats(10)

#We have developed this code using Visual Studio
#Command to run the code: python Time_Analysis_code_Team_5.py

#When we check the time taken by each function using 'cProfile’ and ’pstats’ packages 
#for profiling and analyzing the execution time of our code, we found that "mutual information" feature selection method
#takes the highest time to execute due to computations within the mutual_info_classif 
#function and its internal functions (_compute_mi and _estimate_mi).

#The Code pasted in the main function has only k value = 100.
#Since we want to find the feature selection method that is taking highest time,
# we did not consider keeping the lines of the code related to evaluating the model 
#for multiple k values i.e, k= [50,100,150,200,250].

