#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# In[2]:


seed=pd.read_csv('D://seeds-width-vs-length.csv')
seed


# In[3]:


seed=seed.to_numpy()
correlaton,_=pearsonr(seed[:,0],seed[:,1])
correlaton


# In[5]:


plt.plot(seed[:,0],seed[:,1],'o')
plt.xlabel('width')
plt.ylabel('lenght')
plt.show()


# In[8]:


from sklearn.decomposition import PCA


# In[11]:


pca=PCA()
pca.fit(seed)
transformed=pca.transform(seed)


# In[12]:


mean=pca.mean_
mean


# In[14]:


fpc=pca.components_[0]
fpc


# In[16]:


plt.scatter(seed[:,0],seed[:,1])
plt.arrow(mean[0],mean[1],fpc[0],fpc[1],color='red',width=0.01)
plt.show()


# In[17]:


plt.scatter(transformed[:,0],transformed[:,1])
plt.show()


# In[18]:


from sklearn.datasets import load_iris


# In[26]:


iris=load_iris()


# In[28]:


iris_model=PCA()
iris_model.fit(iris.data[:,[0,2]])
iris_transformed=iris_model.transform(iris.data[:,[0,2]])
mean=iris_model.mean_
fpc=iris_model.components_[0]


# In[30]:


plt.scatter(iris.data[:,0],iris.data[:,2])
plt.arrow(mean[0],mean[1],fpc[0],fpc[1],color='red',width=0.1)
plt.show()


# In[31]:


plt.scatter(iris_transformed[:,0],iris_transformed[:,1])
plt.show()


# In[ ]:




