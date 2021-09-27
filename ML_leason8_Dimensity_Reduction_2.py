#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
iris=load_iris()


# In[2]:


from mpl_toolkits.mplot3d import Axes3D


# In[3]:


fig=plt.figure()
ax=Axes3D(fig)
versicolor=iris.data[50:100]
versicolor=versicolor[:,[0,1,3]]
ax.scatter(versicolor[:,0],versicolor[:,1],versicolor[:,2],c='red')
ax.set_title('iris versicolor')
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
ax.set_zlabel('petal width')
plt.show()


# In[4]:


vc_pca=PCA()
vc_pca.fit(versicolor)
vc_pca_transformed=vc_pca.transform(versicolor)


# In[5]:


nfeaturs=range(vc_pca.n_components_)
var_featcurs=vc_pca.explained_variance_


# In[6]:


plt.bar(nfeaturs,var_featcurs)
plt.xlabel('pca featcure')
plt.ylabel('variance')
plt.xticks(nfeaturs)
plt.show()


# In[7]:


dim_r=PCA(n_components=2)
dim_r.fit(iris.data)
dim_r_transformed=dim_r.transform(iris.data)
dim_r_transformed


# In[8]:


plt.scatter(dim_r_transformed[:,0],dim_r_transformed[:,1],c=iris.target)
plt.show()


# In[ ]:




