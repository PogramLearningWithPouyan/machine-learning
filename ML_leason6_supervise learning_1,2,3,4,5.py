#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


iris =datasets.load_iris()


# In[5]:


iris.data.shape


# In[6]:


iris.feature_names


# In[7]:


iris.target_names


# In[6]:


iris.DESCR


# In[7]:


iris.data


# In[8]:


iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
iris_df


# In[9]:


iris_df['target']=iris.target
iris_df


# In[11]:


pd.plotting.scatter_matrix(iris_df, c=iris.target,s=150,figsize=[11,11])


# In[20]:


x=iris.data
y=iris.target
x


# In[13]:


y


# In[15]:


plt.scatter(x[:,0],x[:,1],c=y)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier


# In[19]:


knn=KNeighborsClassifier(n_neighbors=6,metric='minkowski',p=2)


# In[21]:


knn.fit(x,y)


# In[22]:


y_predict=knn.predict(np.array([[5,3,1,0.2]]))


# In[23]:


y_predict


# In[68]:


from sklearn.model_selection import train_test_split


# In[95]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,shuffle=True)


# In[96]:


x_train.shape


# In[97]:


x_test.shape


# In[1]:


knn=KNeighborsClassifier(n_neighbors=9,metric='minkowski',p=2)


# In[111]:


knn.fit(x_train,y_train)


# In[112]:


knn.score(x_test,y_test)


# In[ ]:





# In[ ]:




