#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# In[2]:


iris=load_iris()
x=iris.data


# In[3]:


from sklearn.cluster import DBSCAN


# In[12]:


dbscan=DBSCAN(eps=0.5,min_samples=20)
dbscan.fit(x)
labels=dbscan.labels_


# In[13]:


plt.scatter(x[:,1],x[:,2],c=labels)
plt.show()


# In[ ]:




