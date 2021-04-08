# -*- coding: utf-8 -*-
"""
@author: Asma Baccouche
"""

import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

table = str.maketrans('', '', string.punctuation)

d = pd.read_csv('english_stopwords.txt', sep=" ", header=None)
stop = [a for a in d[0]]
stop_words1 = set(stopwords.words('english'))
stop_words = list(stop_words1.union(stop))
wordnet_lemmatizer = WordNetLemmatizer()
Numbers = ["0","1","2","3","4","5","6","7","8","9"]

def read_data(path1, path2):
    publications1 = pd.read_csv(path1)
    publications1 = publications1[['USERNAME', 'TITLE']]
    publications2 = pd.read_csv(path2, header=None)
    publications2 = publications2[[1, 9]]
    publications2.columns = ["USERNAME", "TITLE"]
    publications = pd.concat([publications1, publications2])
    
    admin = pd.read_csv('ADMIN.CSV')
    admin.loc[38, 'DEPT_ACAD'] = 'Computer Engineering and Computer Science'
    admin.loc[224, 'DEPT_ACAD'] = 'Mechanical Engineering'
    admin.loc[231, 'DEPT_ACAD'] = 'Industrial Engineering'
    admin.loc[303, 'DEPT_ACAD'] = 'Electrical and Computer Engineering'
    admin.loc[313, 'DEPT_ACAD'] = 'Computer Engineering and Computer Science'
    admin.loc[371, 'DEPT_ACAD'] = 'Chemical Engineering'
    admin.loc[408, 'DEPT_ACAD'] = 'Electrical and Computer Engineering'
    admin.loc[464, 'DEPT_ACAD'] = 'Mechanical Engineering'
    admin.loc[526, 'DEPT_ACAD'] = 'Computer Engineering and Computer Science'
    admin.loc[696, 'DEPT_ACAD'] = 'Mechanical Engineering'
    admin.loc[709, 'DEPT_ACAD'] = 'Industrial Engineering'
    admin.loc[783, 'DEPT_ACAD'] = 'Mechanical Engineering'
    admin.loc[847, 'DEPT_ACAD'] = 'Electrical and Computer Engineering'
    admin.loc[945, 'DEPT_ACAD'] = 'Industrial Engineering'
    admin.loc[961, 'DEPT_ACAD'] = 'Electrical and Computer Engineering'
    admin.loc[1030, 'DEPT_ACAD'] = 'Mechanical Engineering'
    
    publications_admin = publications.merge(admin, on='USERNAME')   
    data = publications_admin[['TITLE', 'DEPT_ACAD']]
    #data = data.drop_duplicates('TITLE').reset_index(drop=True)

    data.loc[data.DEPT_ACAD == 'Bioengineering', 'DEPT_ACAD'] = 'BE'
    data.loc[data.DEPT_ACAD == 'Chemical Engineering', 'DEPT_ACAD'] = 'CHE'
    data.loc[data.DEPT_ACAD == 'Civil and Environmental Engineering', 'DEPT_ACAD'] = 'CE'
    data.loc[data.DEPT_ACAD == 'Computer Engineering and Computer Science', 'DEPT_ACAD'] = 'CECS'
    data.loc[data.DEPT_ACAD == 'Electrical and Computer Engineering', 'DEPT_ACAD'] = 'EE'
    data.loc[data.DEPT_ACAD == 'Engineering Fundamentals', 'DEPT_ACAD'] = 'EF'
    data.loc[data.DEPT_ACAD == 'Industrial Engineering', 'DEPT_ACAD'] = 'IE'
    data.loc[data.DEPT_ACAD == 'Mechanical Engineering', 'DEPT_ACAD'] = 'ME'

    return(data)
        
def strip(s):
    words = s.split()
    s = ' '.join(word.lower() for word in words)
    words = s.split()
    s = ' '.join(w.translate(table) for w in words)
    words = s.split()
    s = ' '.join(wordnet_lemmatizer.lemmatize(word) for word in words)
    a = s.replace('—', "")
    a = a.replace('−', "")
    a = a.replace('–', "")
    a = a.replace('δ', "")
    a = a.replace('β', "")
    a = a.replace('ß', "")
    a = a.replace('α', "")
    a = a.replace('“', "")
    a = a.replace('”', "")
    a = a.replace('\'', "")
    a = a.replace('-', "")
    a = a.replace(',', "")
    a = a.replace('\"', "")
    a = a.replace('’', "")
    words = a.split()
    new = ' '.join(w for w in words if not w in stop_words)
    S = new.split()
    result =  [el for el in S if el not in Numbers]    
    return(' '.join(result))
     
def split_number_word(sentence):
    words = sentence.split()
    new_words=[]
    for word in words:
        if len(word) > 2:
            new=''
            i=0
            while(i<len(word)-1):
                if (word[i] in Numbers and word[i+1] in Numbers) or (word[i] not in Numbers and word[i+1] not in Numbers):
                    new=new+word[i]
                else :
                    if (word[i] in Numbers and word[i+1] not in Numbers) or (word[i+1] in Numbers and word[i] not in Numbers):
                        new=new+word[i]+' '           
                i=i+1
            word=new+word[len(word)-1]
        new_words.append(word)
    return(' '.join(new_words))
    
def strip_if_all_number(sentence):
    words = sentence.split()
    new_words=[]
    flag=False
    count_flag=0
    for word in words:
        i=0
        while(i<len(word)):
            if word[i] in Numbers:
                count_flag=count_flag+1
                flag=True
            i=i+1
        if not(flag is True and count_flag == len(word)):
            new_words.append(word)
               
    return(' '.join(new_words))    

def strip_if_words_number(sentence):
    words = sentence.split()
    new_words=[]
    for word in words:
        flag=False
        count_flag=0
        i=0
        while(i<len(word)-1):
            if word[i] in Numbers and word[i+1] in Numbers:
                count_flag=count_flag+1
                flag=True
            i=i+1
        if word[len(word)-1] in Numbers:
            count_flag=count_flag+1
            flag=True
        else:
            flag=False
                
        if not(flag is True and count_flag == len(word)):
            new_words.append(word)
               
    return(' '.join(new_words))
    
def clean_data(data):
    data = data.rename(columns={"DEPT_ACAD": "Department"})
    for i in range(len(data)):
        a1 = strip(data.loc[i, 'TITLE'])
        a2 = split_number_word(a1)
        a3 = strip_if_all_number(a2)
        data.loc[i, 'TITLE'] = strip_if_words_number(a3)
    d = data.drop_duplicates(subset=['TITLE']).reset_index(drop=True)
    return(d)
    
def plot_wordcloud(DF):
    stopwords2 = set(STOPWORDS)
    stopwords2.update(["using", "based", "new"])
    titles = " ".join(title for title in DF.TITLE)
    print ("There are {} words in the combination of all review.".format(len(titles)))
    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords2, background_color="white").generate(titles)
    # Display the generated image:
    fig = plt.figure(figsize=(10,10))
    ax0 = fig.add_subplot(111)
    ax0.imshow(wordcloud, interpolation='bilinear')
    ax0.axis("off")
    plt.show()
    
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw