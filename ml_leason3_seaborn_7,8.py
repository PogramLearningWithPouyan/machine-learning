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


# In[5]:


sb.boxplot(x='Company',y='Ram',data=smartphones)
plt.show()


# In[10]:


sb.jointplot(x='Capacity',y='Ram',data=smartphones,kind='scatter')
plt.show()


# In[3]:


sb.pairplot(smartphones,hue='Name',palette='hls',plot_kws={'s':80})
plt.show()


# In[ ]:




