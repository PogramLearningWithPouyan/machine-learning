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


from sklearn.cluster import MeanShift


# In[15]:


ms=MeanShift(bandwidth=5)
ms.fit(x)
labels=ms.labels_
center=ms.cluster_centers_


# In[16]:


plt.scatter(x[:,0],x[:,2],c=labels)
plt.scatter(center[:,0],center[:,2],marker='x',s=150,linewidths=5)
plt.show()


# In[ ]:




