#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as ab


# In[2]:


smartphones=pd.read_csv('d://smartphones.csv')
smartphones


# In[4]:


counts=smartphones.Ram.value_counts()


# In[5]:


category=counts.index
category


# In[6]:


plt.bar(category,counts)
plt.xlabel('Ram')
plt.ylabel('counts')
plt.xticks([1,2,3,4])
plt.yticks([1,2,3])
plt.show()


# In[11]:


def ECDF(data):
    n=len(data)
    x=np.sort(data)
    y=np.arange(1,n+1)/n
    return x,y
x,y=ECDF(smartphones.inch)
x


# In[12]:


y


# In[14]:


plt.figure(figsize=(10,7))
plt.scatter(x,y,s=80)
plt.margins(0.05)
plt.xlabel('inch',fontsize=15)
plt.ylabel('ECDF',fontsize=15)
plt.show()


# In[15]:


np.mean(smartphones.inch)


# In[16]:


np.median(smartphones.inch)


# In[18]:


np.percentile(smartphones.inch,[25,50,75])


# In[19]:


np.var(smartphones.inch)


# In[20]:


np.std(smartphones.inch)


# In[ ]:




