#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[2]:


iris =datasets.load_iris()


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3,random_state=42,stratify=iris.target)


# In[5]:


knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)


# In[18]:


neighbors=np.arange(1,30)
train_accuracy=np.empty(len(neighbors))
test_accuracy=np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    knn_model=KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(x_train,y_train)
    train_accuracy[i]=knn_model.score(x_train,y_train)
    test_accuracy[i]=knn_model.score(x_test,y_test)   
plt.plot(neighbors,train_accuracy,label='train accuracy')
plt.plot(neighbors,test_accuracy,label='test accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[19]:


from sklearn.tree import DecisionTreeClassifier


# In[20]:


dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
predict_dtc=dtc.predict(x_test)


# In[22]:


from sklearn import metrics


# In[23]:


metrics.accuracy_score(y_test,predict_dtc)


# In[24]:


dtc.score(x_test,y_test)


# In[ ]:




