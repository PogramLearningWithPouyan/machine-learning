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


from scipy.cluster.hierarchy import linkage,dendrogram,fcluster


# In[7]:


hirarachical=linkage(x,method='complete')
dendrogram(hirarachical)
plt.show()


# In[10]:


labels=fcluster(hirarachical,3,criterion='distance')
labels


# In[11]:


plt.scatter(x[:,0],x[:,2],c=labels)
plt.show()


# In[ ]:




