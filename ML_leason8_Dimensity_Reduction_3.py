#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline


# In[3]:


doc = ['you are watching machine learning course',
       'word frequency array is a part of unsupervised learning from machine learning course',
       'PLP is an online educational channel in youtube']
titles = ['first doc', 'second doc', 'third doc']


# In[4]:


tfidf=TfidfVectorizer()
csr_mat=tfidf.fit_transform(doc)


# In[5]:


word=tfidf.get_feature_names()


# In[7]:


csr_mat.toarray()


# In[11]:


tfidf.vocabulary_


# In[12]:


svd_test=TruncatedSVD(n_components=2)
kmeans_test=KMeans(n_clusters=2)
pipeline=make_pipeline(svd_test,kmeans_test)
pipeline.fit(csr_mat)
labels=pipeline.predict(csr_mat)


# In[14]:


df=pd.DataFrame({'labels':labels,'docs':titles})
df.sort_values('labels')


# In[ ]:




