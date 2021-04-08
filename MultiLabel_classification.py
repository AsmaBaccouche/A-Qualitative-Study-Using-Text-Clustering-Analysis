#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Asma Baccouche
"""

import numpy as np
import pandas as pd
from Data_helper import sent_vectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, jaccard_similarity_score, coverage_error, f1_score
from gensim.models import Word2Vec

random_state = np.random.RandomState(0)

pub_admin = pd.read_csv('admin_publications.csv')

data2 = pub_admin.drop_duplicates(subset=['TITLE', 'Department']).reset_index(drop=True)
data2 = data2.dropna()

grouped = data2.groupby('TITLE')
d = grouped['Department'].aggregate(lambda x: list(x)).reset_index(name="Department")

dd = data2.groupby('TITLE').agg(lambda x: x.tolist())
ts = pd.Series(dd['Department'].values)

p1 = pd.DataFrame(dd.index.tolist(), columns=['TITLE'])
p2 = pd.get_dummies(ts.apply(pd.Series).stack()).sum(level=0)
data =  pd.concat([p1, p2], axis=1)

X = data['TITLE']
y = data[['BE', 'CE', 'CECS', 'CHE', 'EE', 'EF', 'IE', 'ME']]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=random_state)
 
#our pipeline transforms our text into a vector and then applies OneVsRest using LinearSVC
pipeline = Pipeline([('tfidf', TfidfVectorizer()),('clf', OneVsRestClassifier(LinearSVC()))])
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

print(hamming_loss(y_test,y_pred))
print(jaccard_similarity_score(y_test, y_pred))
print(coverage_error(y_test, y_pred))
print(f1_score(y_test, y_pred, average='micro'))
print(f1_score(y_test, y_pred, average='macro'))

sentences = [sentence.split() for sentence in data['TITLE']]
model = Word2Vec(sentences, min_count=1, size=4000)
X_w2v = np.zeros((len(sentences), 4000))
for i in range(len(sentences)):
    X_w2v[i] = sent_vectorizer(sentences[i], model)

scaler = MinMaxScaler()
scaler.fit(X_w2v)
scaler.transform(X_w2v)
X_w2v = scaler.transform(X_w2v)
X_train, X_test, y_train, y_test = train_test_split(X_w2v, y, test_size=.2,random_state=random_state)
    
clf_w2v = OneVsRestClassifier(LinearSVC())
clf_w2v.fit(X_train,y_train)
y_pred = clf_w2v.predict(X_test)
 
print(hamming_loss(y_test,y_pred))
print(jaccard_similarity_score(y_test, y_pred))
print(coverage_error(y_test, y_pred))
print(f1_score(y_test, y_pred, average='micro'))
print(f1_score(y_test, y_pred, average='macro'))