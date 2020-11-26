#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


smartphones=pd.read_csv('d://smartphones.csv')
smartphones


# In[3]:


from scipy.stats import chi2_contingency


# In[4]:


table_ob=pd.crosstab(smartphones.Capacity,smartphones.Ram)
table_ob


# In[5]:


chi , p_value , dof ,table_ex=chi2_contingency(table_ob.values)


# In[6]:


chi


# In[7]:


p_value


# In[8]:


dof


# In[9]:


table_ex


# In[26]:


#np.random.seed(42)
rand_np=np.random.random(3)
rand_np


# In[22]:


win=rand_np>0.5
win


# In[30]:


rand_num=np.random.random(10000)
w=rand_num>0.5
num_head=np.sum(w)/10000
num_head


# In[34]:


sample=np.random.normal(0,1,size=10000)


# In[35]:


sb.distplot(sample)
plt.show()


# In[ ]:




