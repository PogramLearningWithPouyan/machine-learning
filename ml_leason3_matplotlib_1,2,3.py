#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


years=[1960,1970,1980,1990,2000,2010,2020]
iranPop=[21.19,28.51,38.67,56.23,66.13,74.75,80.29]


# In[27]:


plt.figure(figsize=(5,4),dpi=80)
plt.plot(years,iranPop)
plt.xlabel('years')
plt.ylabel('population')
plt.yticks([21.19,28.51,38.67,56.23,66.13,74.75,80.29],['21m','28m','38m','56m','66m','74m','80m'])
plt.show()


# In[6]:


plt.scatter(years,iranPop)
plt.show()


# In[7]:


cityName = ['Tehran','Mashhad','Isfahan','Karaj','Tabriz','Shiraz']
cityPop = [7153309,2307177,1547164,1448075,1424641,1249942]


# In[19]:


plt.hist(cityPop,bins='auto')
plt.show()


# In[21]:


plt.pie(cityPop,labels=cityName)
plt.show()


# In[33]:


popsize=np.array([7153309,2307177,1547164,1448075,1424641,1249942])/10000
colors=['violet','green','orange','tomato','blue','pink']
plt.scatter(np.arange(6),cityPop,s=popsize,c=colors)
plt.margins(0.1)
plt.title('iran polution')
plt.xticks([0,1,2,3,4,5],['Tehran','Mashhad','Isfahan','Karaj','Tabriz','Shiraz'])
plt.yticks([1000000,2000000,3000000,7000000],['1m','2m','3m','7m'])
plt.text(0,7153309,'iran capital',fontsize=15)
plt.show()


# In[ ]:




