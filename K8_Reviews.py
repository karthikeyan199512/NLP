# -*- coding: utf-8 -*-
"""
@author: Karthik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import os
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

os.chdir('D:\\OneDrive\\Manipal\\GitRepo and sagemaker course\\Natural-Language-Processing-master')
reviews = pd.read_csv('K8 Reviews v0.2.csv')
stop_nltk = stopwords.words('english')
stop_updated = stop_nltk + ['mobile','phone','lenovo','k8','note'] # -> domain specific stopwords
stemmer_s = SnowballStemmer('english')

analyser = SentimentIntensityAnalyzer()
def get_vader_sentiment(sent):
    return analyser.polarity_scores(sent)['compound']

reviews['sent_score_vader'] = reviews['review'].apply(get_vader_sentiment)
reviews['sent_prediction_vader'] = reviews['sent_score_vader'].apply(lambda x:1 if x>0.3 else 0)

def clean_txt(sent):
    #Stripping white spaces before and after the text
    sent = sent.strip()
    #Replacing multiple spaces with a single space
    result = re.sub('\s+',' ',sent)
    #Replacing the non-alphanumeric and non space chars with nothing
    result1 = re.sub('[^\s\w]+','',result)
    #Normalize case, stemm and remove shorter tokens
    tokens = word_tokenize(result1.lower())
    stemmed = [stemmer_s.stem(term) for term in tokens if term not in stop_updated and len(term)>2]
    #Join all to form a single string which will be returned from the UDF
    res = ' '.join(stemmed)
    return res

reviews['clean_review'] = reviews.review.apply(clean_txt)
X_text = reviews['clean_review'].values
y = reviews['sent_prediction_vader'].values
X_train,X_test,y_train,y_test = train_test_split(X_text,y,test_size=0.2,random_state=42)
tfidf_vect = TfidfVectorizer()
X_train = tfidf_vect.fit_transform(X_train)
X_test = tfidf_vect.transform(X_test)
pickle.dump(tfidf_vect, open('tranform.pkl', 'wb'))
lr_clf = LogisticRegression()
lr_clf.fit(X_train,y_train)
filename = 'nlp_model.pkl'
pickle.dump(lr_clf,open(filename,'wb'))
y_pred = lr_clf.predict(X_test)
accuracy_score(y_test,y_pred)
