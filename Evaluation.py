# -*- coding: utf-8 -*-
"""
@author: Asma Baccouche
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from Get_Data import get_data
from sklearn.cluster import KMeans, SpectralClustering
from nltk.cluster import KMeansClusterer, util
from gensim.models import Word2Vec
from sklearn import metrics
import numpy as np
from Data_helper import sent_vectorizer

data, deps, Deps_Count = get_data()
sentences = [sentence for sentence in data['TITLE']]
s = [sentence.split() for sentence in data['TITLE']]

vectorizer1 = CountVectorizer()
vectorizer2 = TfidfVectorizer()

X_TF = vectorizer1.fit_transform(sentences)
X_TFIDF = vectorizer2.fit_transform(sentences)

kmeans1 = KMeans(n_clusters=8).fit(X_TF)
labels1 = kmeans1.labels_

kmeans2 = KMeans(n_clusters=8).fit(X_TFIDF)
labels2 = kmeans2.labels_

SpectralClustering1 = SpectralClustering(n_clusters=8, assign_labels="discretize", random_state=0).fit(X_TF)
labels3 = SpectralClustering1.labels_

SpectralClustering2 = SpectralClustering(n_clusters=8, assign_labels="discretize", random_state=0).fit(X_TFIDF)
labels4 = SpectralClustering2.labels_

model = Word2Vec(s, min_count=1)
X = np.zeros((len(s), 100))
for i in range(len(s)):
    X[i] = sent_vectorizer(s[i], model)

kclusterer = KMeansClusterer(8, distance=util.cosine_distance, repeats=25)
labels5 = kclusterer.cluster(X, assign_clusters=True)

print("Silhouette score : %0.3f" % metrics.silhouette_score(X_TF, labels1, metric='euclidean'))
print("Silhouette score : %0.3f" % metrics.silhouette_score(X_TFIDF, labels2, metric='euclidean'))
print("Silhouette score : %0.3f" % metrics.silhouette_score(X_TF, labels3, metric='euclidean'))
print("Silhouette score : %0.3f" % metrics.silhouette_score(X_TFIDF, labels4, metric='euclidean'))
print("Silhouette score : %0.3f" % metrics.silhouette_score(X, labels5, metric='euclidean'))
