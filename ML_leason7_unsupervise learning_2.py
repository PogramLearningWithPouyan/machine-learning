#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[4]:


iris=load_iris()
x=iris.data
x


# In[6]:


kmn=KMeans(n_clusters=3)
kmn.fit(x)
labels=kmn.predict(x)
labels


# In[7]:


center=kmn.cluster_centers_


# In[8]:


plt.scatter(x[:,0],x[:,2],c=labels)
plt.scatter(center[:,0],center[:,2],marker='x',s=150)
plt.show()


# In[9]:


kmn.inertia_


# In[10]:


y=[]
for k in np.arange(1,6):
    kmn=KMeans(n_clusters=k)
    kmn.fit(x)
    y.append(kmn.inertia_)
y


# In[11]:


plt.plot(np.arange(1,6),y,'o-')
plt.xlabel('number of clusters')
plt.ylabel('inertia')
plt.show()


# In[ ]:




