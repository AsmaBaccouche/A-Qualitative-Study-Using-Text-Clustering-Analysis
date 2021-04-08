# -*- coding: utf-8 -*-
"""
@author: Asma Baccouche
"""
from Data_helper import sent_vectorizer
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np
import pandas as pd
from sklearn import cluster, metrics
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
  

pub_admin = pd.read_csv('admin_publications.csv')

data2 = pub_admin.drop_duplicates(subset=['TITLE', 'Department']).reset_index(drop=True)
data2 = data2.dropna()

d = data2.groupby('TITLE').agg(lambda x: x.tolist())

ts = pd.Series(d['Department'].values)

p1 = pd.DataFrame(d.index.tolist(), columns=['TITLE'])

p2 = pd.get_dummies(ts.apply(pd.Series).stack()).sum(level=0)

data =  pd.concat([p1, p2], axis=1)

deps = data[['BE', 'CE', 'CECS', 'CHE', 'EE', 'EF', 'IE', 'ME']]


Count = np.zeros((8, 8))
for col1 in deps.columns.tolist():
    for col2 in deps.columns.tolist():
        Count[deps.columns.tolist().index(col1)][deps.columns.tolist().index(col2)] = len([i for i, j in zip(deps[col1], deps[col2]) if i == j == 1])

Deps_Count = pd.DataFrame(data=Count,index =deps.columns.tolist(), columns=deps.columns.tolist())



sentences = [sentence.split() for sentence in data['TITLE']]
model = Word2Vec(sentences, min_count=1)

X=[]
for sentence in sentences:
    X.append(sent_vectorizer(sentence, model))   


NUM_CLUSTERS=8
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
  
labels_nltk = assigned_clusters
 
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)
  
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
    
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(X))
  
silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
  
print ("Silhouette_score: ")
print (silhouette_score)

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
 
Y=model.fit_transform(X)
  
plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290,alpha=.5)
 
 
for j in range(len(sentences)):    
   plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
   print ("%s %s" % (assigned_clusters[j],  sentences[j]))
 
plt.show()

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(deps)
labels = kmeans.labels_
print(metrics.silhouette_score(deps, labels, metric='euclidean'))
print (kmeans.score(deps))