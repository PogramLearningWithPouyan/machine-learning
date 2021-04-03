#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[4]:


from sklearn import datasets


# In[5]:


bcd= datasets.load_breast_cancer()
x=bcd.data
y=bcd.target
x.shape


# In[6]:


bcd.target_names


# In[ ]:





# In[7]:


y


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[9]:


knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)


# In[10]:


from sklearn.metrics import confusion_matrix, classification_report


# In[12]:


print(confusion_matrix(y_test,y_pred,[0,1]))
print(classification_report(y_test,y_pred))


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[19]:


lr=LogisticRegression()
lr.fit(x_train, y_train)
y_pred=lr.predict(x_test)
cm=confusion_matrix(y_test,y_pred,[0,1])
cm


# In[20]:


from sklearn.preprocessing import normalize


# In[21]:


cm=normalize(cm,norm='l1',axis=1)


# In[22]:


cm_df=pd.DataFrame(cm,columns=bcd.target_names,index=bcd.target_names)
cm_df


# In[23]:


from sklearn.metrics import roc_curve


# In[26]:


y_pred_prob=lr.predict_proba(x_test)[:,1]
fpr,tpr,thresholds=roc_curve(y_test,y_pred_prob)


# In[27]:


plt.plot([0,1],[0,1],'k--')


# In[29]:


plt.plot(fpr,tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()


# In[30]:


from sklearn.metrics import roc_auc_score


# In[31]:


roc_auc_score(y_test,y_pred_prob)


# In[ ]:




