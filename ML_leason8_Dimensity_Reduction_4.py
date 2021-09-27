#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


doc = ['you are watching machine learning course',
       'word frequency array is a part of unsupervised learning from machine learning course',
       'PLP is an online educational channel in youtube']
titles = ['first doc', 'second doc', 'third doc']


# In[3]:


tfidf=TfidfVectorizer()
csr_mat=tfidf.fit_transform(doc)


# In[4]:


from sklearn.decomposition  import NMF


# In[5]:


nmf=NMF(n_components=2)
nmf.fit(csr_mat)
nmf_transformed=nmf.transform(csr_mat)


# In[6]:


pd.DataFrame(nmf.components_,columns=sorted(tfidf.vocabulary_))


# In[7]:


nmf_transformed


# In[8]:


nmf_df=pd.DataFrame(nmf_transformed,index=titles)
nmf_df


# In[9]:


mat=[[1,2,3],[4,5,6],[7,8,9]]
mat_nmf=NMF(n_components=2)
mat_nmf.fit(mat)
nmf_features=mat_nmf.transform(mat)
nmf_features


# In[10]:


mat_nmf.components_


# In[11]:


np.dot(nmf_features,mat_nmf.components_)


# In[ ]:




