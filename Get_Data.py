# -*- coding: utf-8 -*-
"""
@author: Asma Baccouche
"""

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def get_data():
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
    
    return data, deps, Deps_Count


def prepare_text(df):
    text = " ".join(df.TITLE)
    noise_words = ['brief', 'say', 'update', 'ing']
    for noise in noise_words:
        text = text.lower().replace(noise," ")
    return text

text = prepare_text(data)
wordcloud = WordCloud(background_color='white').generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure(figsize=(140,140))
plt.show()

