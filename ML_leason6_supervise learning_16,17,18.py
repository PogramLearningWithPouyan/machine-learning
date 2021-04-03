#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


from sklearn.model_selection import GridSearchCV


# In[11]:


from sklearn import datasets
bcd= datasets.load_breast_cancer()
x=bcd.data
y=bcd.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[6]:


p_grid={'n_neighbors':np.arange(1,50)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,p_grid,cv=5)
knn_cv.fit(x,y)
print(knn_cv.best_params_)
print(knn_cv.best_score_)


# In[7]:


from scipy.stats import randint


# In[8]:


randint(1,9).rvs(3)


# In[9]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier


# In[14]:


params={'max_depth':[None,3],'max_features':randint(1,9),'min_samples_leaf':randint(1,9)}
tree=DecisionTreeClassifier()
tree_cv=RandomizedSearchCV(tree,params,cv=10)
tree_cv.fit(x_train,y_train)
print(tree_cv.best_params_)
print(tree_cv.best_score_)


# In[15]:


score=tree_cv.score(x_test,y_test)
score


# In[16]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


gnb=GaussianNB()
y_pred=gnb.fit(x_train,y_t)

