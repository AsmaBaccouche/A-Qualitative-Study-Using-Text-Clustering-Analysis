# -*- coding: utf-8 -*-
"""
@author: Asma Baccouche
"""

import numpy as np
from Data_helper import sent_vectorizer
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from scipy.stats import multivariate_normal
from gensim.models import Word2Vec
import networkx as nx
import string
import matplotlib.pyplot as plt
from Get_Data import get_data

data, deps, Deps_Count = get_data()

sentences = [sentence.split() for sentence in data['TITLE']]
max_length = max([len(sentence) for sentence in sentences])

model = Word2Vec(sentences, min_count=1, size=max_length)
X = np.zeros((len(sentences), max_length))
for i in range(len(sentences)):
    X[i] = sent_vectorizer(sentences[i], model)


y = data[['BE', 'CE', 'CECS', 'CHE', 'EE', 'EF', 'IE', 'ME']]
n_clusters = 8
 
gmm = GaussianMixture(n_components=n_clusters).fit(X)
print('Converged:',gmm.converged_) # Check if the model has converged
means = gmm.means_ 
covariances = gmm.covariances_
weights = gmm.weights_

Thetas=[]
for sentence in sentences:
    theta = np.zeros((n_clusters, 1))
    for i in range(n_clusters):
        a = theta[i]
        for w in sentence:
            a = a + weights[i] * multivariate_normal.pdf(model[w],mean=means[i],cov=covariances[i])        
        theta[i] = a

    theta = theta / np.sum(theta)
    Thetas.append(theta)

D=np.zeros((len(sentences), len(sentences)))
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i!=j:
            dist = np.linalg.norm(Thetas[i]-Thetas[j])
            D[i][j] = dist
            
Words=[]
for sentence in sentences:
    for w in sentence:
        Words.append(w)
        
Words = [x for x in Words if x != 'a']
        
Word_label=[]
for w in Words:
    topics_label=np.zeros((n_clusters, 1))
    for i in range(n_clusters):
        pi = weights[i] * multivariate_normal.pdf(model[w],mean=means[i],cov=covariances[i])
        topics_label[i] = pi
    topics_label = topics_label / np.sum(topics_label)   
    Word_label.append(topics_label)
    
Labels={}
for i in range(len(Words)):
    Labels[Words[i]] = np.argmax(Word_label[i])+1
    

data_new = gmm.sample(100)


# Visualization
G = nx.from_numpy_matrix(D)
pos=nx.spring_layout(G)
nx.draw(G,pos,node_color='#A0CBE2',edge_color='#BB0000',width=2,edge_cmap=plt.cm.Blues,with_labels=False)
plt.savefig("graph.png", dpi=1000)

G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),string.ascii_uppercase)))
from networkx.drawing.nx_agraph import graphviz_layout
pos = graphviz_layout(G)
from networkx.drawing.nx_agraph import write_dot
write_dot(G, pos)

nx.draw(G, pos)
G = nx.drawing.nx_agraph.to_agraph(G)
G.node_attr.update(color="red", style="filled")
G.edge_attr.update(color="blue", width="2.0")
G.draw('/tmp/out.png', format='png', prog='neato')

