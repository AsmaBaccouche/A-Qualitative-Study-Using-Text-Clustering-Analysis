# -*- coding: utf-8 -*-
"""
@author: Asma Baccocuhe
"""

from Data_helper import read_data, clean_data, sent_vectorizer
from gensim.models import Word2Vec
from sklearn import preprocessing
from gensim import corpora, models
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation

path1 = 'Publications.csv'
path2 ='Publications2.csv'
data = read_data(path1, path2)
# Visulize the WordCloud
DF = clean_data(data)

le = preprocessing.LabelEncoder()
le.fit(DF['Department'])
n_classes = len(le.classes_)
DF['True_Class']=le.transform(DF['Department'])
y = DF['True_Class']

lb = preprocessing.LabelBinarizer()
lb.fit(DF['Department'])
n_classes = len(lb.classes_)
y2=lb.fit_transform(DF['Department'])

sentences = [sentence.split() for sentence in DF['TITLE']]
max_length = max([len(sentence) for sentence in sentences])

dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(text) for text in sentences]
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=8, id2word = dictionary, passes=20)

Top_words_topic=[]
for i in range(8):
    Top_words_topic.append(ldamodel.show_topic(i))

topics = ldamodel.get_document_topics(corpus, per_word_topics=True)
all_topics = [(doc_topics, word_topics, word_phis) for doc_topics, word_topics, word_phis in topics]

topic_vectors = np.zeros((len(sentences), 8))
for i in range(len(sentences)):
    j=0
    while(j<len(all_topics[i][0])):
        topic_vectors[i][j] = all_topics[i][0][j][1]
        j=j+1
        
        
X_train,X_test,y_train,y_test=train_test_split(topic_vectors,DF['True_Class'],test_size=0.25,random_state=0) 

clf = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial').fit(X_train,y_train)
y_pred=clf.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

model = Word2Vec(sentences, min_count=1, size=max_length)

X=[]
for sentence in sentences:
    x=sent_vectorizer(sentence, model)
    X.append((x-min(x))/(max(x)-min(x)))
lda = LatentDirichletAllocation(n_components=8, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(X)

topic_vectors_2 = lda.transform(X[:])


sim=metrics.pairwise.cosine_similarity(topic_vectors, topic_vectors_2)