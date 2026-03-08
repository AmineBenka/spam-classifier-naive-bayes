import numpy as np
import re

# Class for detecting spam using Naive Bayes
class SpamDetectorNB:
    
    def __init__(self):
        self.log_priors = {}
        self.word_counts = {}
        self.n_words = {}
        self.n_unique_words = {}

    # Tokenize a text into words    
    def tokenize(self, text):
        return re.split("\\W+", text.lower())
     
    # Count how many times each word appears in a text. 
    # Returns a dictionary that contains for each word the number of times it appears in the text. 
    def count_words(self, text):
        new = self.tokenize(text)
        d = {}
        for i in new:
            if i in d:
                d[i] += 1
            else:
                d[i] = 1
        return d
                
        
    
    # Compute log class priors log(𝑃(ℎ𝑎𝑚)) and log(𝑃(sp𝑎𝑚))  
    # by counting up how many spam/ham messages are in our dataset and dividing by the total number
    def compute_log_priors(self, y_train):
        P_spam = sum(y_train == 'spam') / len(y_train)
        P_ham = sum(y_train == 'ham') / len(y_train)
        log_spam = np.log(P_spam)
        log_ham = np.log(P_ham)
        self.log_priors['spam'] = log_spam
        self.log_priors['ham'] = log_ham
        return log_spam, log_ham
    
    # Compute global word counts in the training set (for spam and ham separately)
    def compute_word_counts(self, X_train, y_train):
        self.word_counts['spam'] = {}
        self.word_counts['ham'] = {}
        for text, label in zip(X_train, y_train):
            word_counts = self.count_words(text)
            for word, count in word_counts.items():
                if word in self.word_counts[label]:
                    self.word_counts[label][word] += count
                else:
                    self.word_counts[label][word] = count
            
                
    # Compute all necessary features to train the model
    def train(self, X_train, y_train):
        self.compute_word_counts(X_train, y_train)
        self.log_priors["spam"], self.log_priors["ham"] = self.compute_log_priors(y_train)
        self.n_words['spam'] = sum(self.word_counts['spam'].values())
        self.n_words['ham'] = sum(self.word_counts['ham'].values())
        self.n_unique_words['spam'] = len(self.word_counts['spam'])
        self.n_unique_words['ham'] = len(self.word_counts['ham'])
        
    def predict(self, X_test):
        predictions = []
        for text in X_test:
            # Compute word counts
            word_counts = self.count_words(text)
            # Initialize log posteriors 𝑙𝑜𝑔(𝑃(spam|message)) and 𝑙𝑜𝑔(𝑃(ham|message)) according to log priors

            P_SsM  = self.log_priors['spam']
            P_HsM = self.log_priors['ham']

            # Update log posteriors with the log likelihood of each word
            for word, count in word_counts.items():
                PsS_word = (self.word_counts['spam'].get(word, 0) + 1)/(self.n_words['spam'] + self.n_unique_words['spam'])
                PsH_word = (self.word_counts['ham'].get(word, 0) + 1)/(self.n_words['ham'] + self.n_unique_words['ham'])
    
                # Update log posteriors
                P_SsM += count*np.log(PsS_word)
                P_HsM += count*np.log(PsH_word)
            
            if P_SsM > P_HsM:
                predictions.append('spam')
            else:
                predictions.append('ham')

        return predictions