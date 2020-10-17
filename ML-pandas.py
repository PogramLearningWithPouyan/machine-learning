#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


mySeries=pd.Series([1,2,3,4,5],index=['row1','row2','row3','row4','row5'])
mySeries


# In[4]:


mySeries.values


# In[5]:


mySeries.index


# In[6]:


mySeries.row2


# In[7]:


mySeries['row2']


# In[8]:


mySeries[mySeries>3]


# In[9]:


mySeries.rename({'row1':'a', 'row2':'b', 'row3':'c', 'row4':'d', 'row5':'e'})


# In[10]:


myArrey=np.array([[1,5,9,13],[2,6,10,14],[3,7,11,15],[4,8,12,16]])
myDf=pd.DataFrame(myArrey,index=['row1', 'row2', 'row3', 'row4']
                  ,columns=['col1','col2','col3','col4'])
myDf


# In[11]:


myDict={'col1':[1,2,3,4],'col2':[5,6,7,8],'col3':[9,10,11,12],'col4':[13,14,15,16]}
myDf2=pd.DataFrame(myDict,index=['row1', 'row2', 'row3', 'row4']
                  ,columns=['col1','col2','col3','col4'])
myDf2


# In[12]:


myDf.index


# In[13]:


myDf.columns


# In[14]:


myDf.values


# In[15]:


myDf.loc['row1']['col2']


# In[16]:


myDf.iloc[0][1]


# In[17]:


myDf['col5']=[20,21,22,23]


# In[18]:


myDf


# In[19]:


myDf.loc[['row1','row2'],'col1']=0


# In[20]:


myDf


# In[21]:


myDf.reset_index(drop=True,inplace=True)
myDf


# In[22]:


myDf.drop('col5',axis=1,inplace=True)
myDf


# In[23]:


myDf.rename(columns={'col4':'c4'},inplace=True)
myDf


# In[26]:


myDf=myDf.replace(0,1)


# In[27]:


myDf


# In[28]:


myDf.col1=['{:.2f}'.format(x) for x in myDf.iloc[:,0]]
myDf


# In[29]:


myDf['col2']=myDf['col2'].apply(lambda x:'{:.2f}'.format(x))
myDf


# In[35]:


myDf.sort_index(axis=0,ascending=False)


# In[36]:


myDf.sort_values(by='col1',ascending=False)


# In[37]:


myDf.head(2)


# In[38]:


myDf.tail(2)


# In[39]:


data=pd.read_csv('D://smartphones.csv')
data


# In[ ]:




