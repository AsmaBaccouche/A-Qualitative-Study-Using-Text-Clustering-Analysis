# -*- coding: utf-8 -*-
"""
@author: Asma Baccouche
"""

from sklearn.cluster import KMeans
from Data_helper import read_data, clean_data, sent_vectorizer
import numpy as np
from gensim.models import Word2Vec
from sklearn import metrics, preprocessing
from collections import Counter 

path1 = 'Publications.csv'
path2 ='Publications2.csv'
data = read_data(path1, path2)
# Visulize the WordCloud
DF = clean_data(data)

le = preprocessing.LabelEncoder()
le.fit(DF['Department'])
classes = ['BE', 'CE', 'CECS', 'CHE', 'EE', 'EF', 'IE', 'ME']
n_classes = len(le.classes_)
DF['True_Class']=le.transform(DF['Department'])

y = DF['True_Class']

sentences = [sentence.split() for sentence in DF['TITLE']]
max_length = max([len(sentence) for sentence in sentences])

d1 = DF[DF['True_Class'] == 0]
d2 = DF[DF['True_Class'] == 1]
d3 = DF[DF['True_Class'] == 2]
d4 = DF[DF['True_Class'] == 3]
d5 = DF[DF['True_Class'] == 4]
d6 = DF[DF['True_Class'] == 5]
d7 = DF[DF['True_Class'] == 6]
d8 = DF[DF['True_Class'] == 7]

topic1 = []
topic2 = []
topic3 = []
topic4 = []
topic5 = []
topic6 = []
topic7 = []
topic8 = []

for sentence in d1['TITLE']:
    for w in sentence.split():
        topic1.append(w)
        
for sentence in d2['TITLE']:
    for w in sentence.split():
        topic2.append(w)
        
for sentence in d3['TITLE']:
    for w in sentence.split():
        topic3.append(w)
        
topic3 = [x for x in topic3 if x != 'a']
       
for sentence in d4['TITLE']:
    for w in sentence.split():
        topic4.append(w)
        
for sentence in d5['TITLE']:
    for w in sentence.split():
        topic5.append(w)
        
for sentence in d6['TITLE']:
    for w in sentence.split():
        topic6.append(w)
        
for sentence in d7['TITLE']:
    for w in sentence.split():
        topic7.append(w)
        
for sentence in d8['TITLE']:
    for w in sentence.split():
        topic8.append(w)
        

reference_words=[]
for topic in [topic1, topic2, topic3, topic4, topic5, topic6, topic7, topic8]:
    a = Counter(topic)
    reference_words.append(a.most_common(10)[0][0])

#model = Word2Vec(sentences, min_count=1, size=max_length)
model = Word2Vec(sentences, min_count=1, size=300)

All_Weights=[]
for i in range(len(sentences)):
    Weights=[]
    for word in sentences[i]:
        ref_vector = model[reference_words[DF.loc[i, 'True_Class']]]
        c = 1 - (np.linalg.norm(model[word]-ref_vector) / np.linalg.norm(ref_vector))
        Weights.append(c)
    All_Weights.append(Weights)
    
New_sentences = np.zeros((len(sentences), 300))
for i in range(len(sentences)):
    C=All_Weights[i]
    sentence = np.zeros(300)
    for j in range(len(C)):
        sentence = sentence + C[j]*model[sentences[i][j]]        
    New_sentences[i] = (sentence/len(C))

kmeans = KMeans(n_clusters=8, random_state=0).fit(New_sentences)
labels1 = kmeans.labels_

print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels1))
print("Completeness: %0.3f" % metrics.completeness_score(y, labels1))
print("V-measure: %0.3f" % metrics.v_measure_score(y, labels1))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(y, labels1))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(y, labels1))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(New_sentences, labels1, metric='sqeuclidean'))


X = np.zeros((len(sentences), 300))
for i in range(len(sentences)):
    X[i] = sent_vectorizer(sentences[i], model)
    
kmeans = KMeans(n_clusters=8, random_state=0).fit(X)
labels2 = kmeans.labels_

print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels2))
print("Completeness: %0.3f" % metrics.completeness_score(y, labels2))
print("V-measure: %0.3f" % metrics.v_measure_score(y, labels2))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(y, labels2))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(y, labels2))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels2, metric='sqeuclidean'))


def identity(a, b):
    if a == b :
        return(1)
    else:
        return(0)

count1=0
count2=0
for i in range(len(X)):
    count1=count1+identity(y[i], labels1[i])
    count2=count2+identity(y[i], labels2[i])
    
acc1 = count1/len(X)
print(acc1)

acc2 = count2/len(X)
print(acc2)

NMI1 = metrics.normalized_mutual_info_score(y, labels1)
print(NMI1)

NMI2 = metrics.normalized_mutual_info_score(y, labels2)
print(NMI2)