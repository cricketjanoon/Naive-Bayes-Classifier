# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:06:33 2020

@author: cricketjanoon
"""


import pandas as pd
import string
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

#remove links, @username, RT 
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


# df_tweets = pd.read_csv("Tweets.csv")
# df_tweets = preprocess_tweets(df_tweets)
# df_tweets.to_csv("cleaned_tweets.csv", index=False)

cleaned_tweets = pd.read_csv("cleaned_tweets.csv")