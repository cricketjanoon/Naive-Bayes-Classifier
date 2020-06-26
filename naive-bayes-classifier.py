# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:06:33 2020

@author: cricketjanoon
"""

from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix, precision_score, accuracy_score, f1_score, recall_score
# sklearn metrics used only to compare results

import pandas as pd
import numpy as np
import string
import math
import re


def deEmojify(text):
    '''
    Removes emojis from the string.
    # Link: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    '''
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def clean_links_usernames(text):
    '''
    Removes http links and twitter usernames like @cricketjanoon
    '''
    return " ".join([word for word in text.split() if 'http' not in word and '@' not in word and '<' not in word])

def remove_punctuation(text):
    """
    Removes puntutaions given in in string.punctuations
    """
    words = text.split()
    table = str.maketrans("", "", string.punctuation)
    return ' '.join([w.translate(table) for w in words])

def remove_stopwords(text):
    """
    Removes stopwords taken from NLTK website.
    """
    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    return " ".join([word for word in text.split() if word not in stopwords])

def remove_single_char(text):
    """
    Removes any single character words like 'c', 'w', 'i' etc
    """
    return " ".join([word for word in text.split() if not len(word)==1])

def remove_flight_numbers(text):
    """
    Removes any words containing digits like fligh numbers, phone number etc
    """
    return ' '.join(word for word in text.split() if not any(char.isdigit() for char in word))

def count(text):
    """
    Returns the length of string.
    """
    return len(text)

def preprocess_tweets(df_tweets):
    df_tweets['text'] = df_tweets['text'].apply(deEmojify)
    df_tweets['text'] = df_tweets['text'].apply(clean_links_usernames)
    df_tweets['text'] = df_tweets['text'].apply(lambda x: re.sub('  ', '', x))  
    df_tweets['text'] = df_tweets['text'].apply(lambda x: x.lower())
    df_tweets['text'] = df_tweets['text'].apply(remove_punctuation)
    df_tweets['text'] = df_tweets['text'].apply(lambda x: x.strip())
    df_tweets['text'] = df_tweets['text'].apply(remove_stopwords)
    df_tweets['text'] = df_tweets['text'].apply(remove_flight_numbers)
    df_tweets['text'] = df_tweets['text'].apply(remove_single_char)
    df_tweets = df_tweets[df_tweets['text'].apply(count) != 0]
    return df_tweets


# un-comment this for running the first time
# df_tweets = pd.read_csv("Tweets.csv")
# df_tweets = preprocess_tweets(df_tweets)
# df_tweets.to_csv("cleaned_tweets.csv", index=False)

cleaned_tweets = pd.read_csv("cleaned_tweets.csv")

# extract unique vocabolary words
vocab = []
for index, row in cleaned_tweets.iterrows():
    vocab += [x for x in row['text'].split() if x not in vocab]
    

# calculating the prior probability
counts = cleaned_tweets['airline_sentiment'].value_counts()
prior_prob = {} 
prior_prob['positive'] = counts['positive'] / cleaned_tweets['airline_sentiment'].count()
prior_prob['negative'] = counts['negative'] / cleaned_tweets['airline_sentiment'].count()
prior_prob['neutral'] = counts['neutral'] / cleaned_tweets['airline_sentiment'].count()


# Stratified splitting of data into training and test data (80:20)
sentiment_wise_tweets = {}
sentiments = ['positive', 'negative', 'neutral']

train_data= pd.DataFrame({'airline_sentiment':'', 'text':''}, index=[])
test_data = pd.DataFrame({'airline_sentiment':'', 'text':''}, index=[])

for sentiment in sentiments:
    cur_tweets = cleaned_tweets[cleaned_tweets['airline_sentiment'] == sentiment]
    sentiment_wise_tweets[sentiment] = cur_tweets
    cur_train_data = cur_tweets.iloc[0:int(cur_tweets.shape[0]*0.8)]
    cur_test_data = cur_tweets.iloc[int(cur_tweets.shape[0]*0.8):cur_tweets.shape[0]]
    train_data = pd.concat([train_data, cur_train_data])
    test_data = pd.concat([test_data, cur_test_data])
    

# extracting sentiment wise vocabolary
sentiment_vocab = {'positive':[], 'negative':[], 'neutral':[]}
for sentiment in sentiments:
    for index, row in sentiment_wise_tweets[sentiment].iterrows():
        sentiment_vocab[sentiment] += [x for x in row['text'].split() if x not in sentiment_vocab[sentiment]]
      
# preparing sentiment-wise word count dictionary
sentiment_vocab_word_count = {'positive':[], 'negative':[], 'neutral':[]}
for sentiment in sentiments:
    sentiment_vocab_word_count[sentiment] = {word:1 for word in vocab} # one for normalizing
    
# counting appearance of each word in each class
for index, tweet in train_data.iterrows():
    for word in tweet['text'].split():
        sentiment_vocab_word_count[tweet['airline_sentiment']][word] +=1
    
# total words appeared in each class (including the one we added for normalizing)
total_words_count = {'positive':0, 'negative':0, 'neutral':0}
for sentiment in sentiments:
    for key, value in sentiment_vocab_word_count[sentiment].items():
        total_words_count[sentiment] += value

#calculating P(word|class) for all three classes
for sentiment in sentiments:
    normalizing_factor = total_words_count[sentiment]
    sentiment_vocab_word_count[sentiment] = {key:value/(normalizing_factor) for key, value in sentiment_vocab_word_count[sentiment].items()}
    
# code to check whether probability for all vocab in each class sums to 1
# for sentiment in sentiments:
#     prob = 0.0
#     for key, value in sentiment_vocab_word_count[sentiment].items():
#         prob += value
#     print("Probability for {}: {}".format(sentiment, prob), '\n')


# testing phase  
label_dict = {'positive': 0, 'negative': 1, 'neutral': 2}    
test_labels = np.zeros((test_data.shape[0], 1), dtype=np.int32)
predicted_labels = np.zeros((test_data.shape[0], 1), dtype=np.int32)

for index, label in enumerate(test_data['airline_sentiment'].to_numpy()):
    test_labels[index] = label_dict[label]

index = 0
for i, test_tweet in test_data.iterrows():
    prob_sent = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}   
    for sentiment in sentiments:
        prob_sent[sentiment] = math.log(prior_prob[sentiment])
        for word in test_tweet['text'].split():
            prob_sent[sentiment] += math.log(sentiment_vocab_word_count[sentiment][word])
    max_key = max(prob_sent, key=lambda k: prob_sent[k])
    predicted_labels[index] = label_dict[max_key]
    index +=1
 
    
# Calculating confusion matrix and evaluation metrics: accuracy, precision, recall, f1-score   
print('-'*5, "My own calculated Metrics and Confusion Matrix", '-'*5) 
confusion_matrix = np.zeros((3,3), dtype=np.int32) # because there are three classes
for i in range(test_data.shape[0]):
    confusion_matrix[test_labels[i][0]][predicted_labels[i][0]] += 1
print(confusion_matrix)
    
true_positive = np.sum(confusion_matrix.diagonal())
recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
accuracy = true_positive / test_data.shape[0]
f1 = (2*np.mean(precision)*np.mean(recall)) / (np.mean(recall) + np.mean(precision))

print("Accuracy:", accuracy)
print("Precision:", np.mean(precision))
print("Recall:", np.mean(recall))
print("F1-Score:", f1)

print('-'*10, "SkLearn Metrics and Confusion Matrix", '-'*10)

sk_confustion_matrix = sklearn_confusion_matrix(test_labels, predicted_labels)
print(sk_confustion_matrix)

sk_accuracy = accuracy_score(test_labels, predicted_labels)
sk_precision = precision_score(test_labels, predicted_labels, average='macro')
sk_recall = recall_score(test_labels, predicted_labels, average='macro')
sk_f1 = f1_score(test_labels, predicted_labels, average='macro')

print('Accuracy:', sk_accuracy)
print('Precision:', sk_precision)
print('Recall:', sk_recall)
print('F1 score:', sk_f1)