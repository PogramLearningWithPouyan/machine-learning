#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


smartphones=pd.read_csv('d://smartphones.csv')
smartphones


# In[3]:


plt.scatter(smartphones.inch,smartphones.Weight,s=80)
plt.margins(0.05)
plt.xlabel('screen size')
plt.ylabel('weight')
plt.show()


# In[4]:


np.mean(smartphones.inch)


# In[5]:


np.mean(smartphones.Weight)


# In[6]:


np.cov(smartphones.inch,smartphones.Weight)


# In[8]:


sb.pairplot(smartphones)
plt.show()


# In[10]:


from scipy.stats import pearsonr


# In[12]:


pearson_coeff , p_value =pearsonr(smartphones.inch,smartphones.Weight)
pearson_coeff


# In[13]:


num_var=smartphones.drop(['Name','OS','Capacity','Ram','Company'],axis=1)
num_var


# In[14]:


cor=num_var.corr()


# In[15]:


cor


# In[17]:


sb.heatmap(cor,xticklabels=cor.columns,yticklabels=cor.columns,vmin=-1,vmax=1)
plt.show()


# In[19]:


categor=smartphones.drop(['Name','OS','Weight','inch','Company'],axis=1)
categor


# In[20]:


from scipy.stats import spearmanr


# In[21]:


spearman_coeff , p_value=spearmanr(categor.Capacity,categor.Ram)
spearman_coeff


# In[ ]:




