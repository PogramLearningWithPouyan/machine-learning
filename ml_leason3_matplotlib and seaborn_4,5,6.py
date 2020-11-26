#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


years=[1960,1970,1980,1990,2000,2010,2020]
iranPop=[21.19,28.51,38.67,56.23,66.13,74.75,80.29]
turkeyPop=[27.47,34.88,43.98,53.92,63.24,72.33,79.51]


# In[24]:


plt.plot(years,iranPop,ls='-',marker='+',mew=8)
plt.plot(years,turkeyPop,ls='--',lw=1)
plt.title('iran us turkey')
plt.legend(['iran','terkey'],loc='best')
plt.xlabel('years')
plt.ylabel('population')
plt.yticks([20,30,40,50,60,70,80],['20m','30m','40m','50m','60m','70m','80m'])
plt.grid()
plt.annotate('iran in 1990',xytext=(1990,40),xy=(1990,56.23),
             arrowprops={'facecolor':'silver','width':4},fontsize=15)
plt.show()


# In[25]:


plt.subplot(1,2,1)
plt.plot(years,iranPop)
plt.title('iran population')
plt.subplot(1,2,2)
plt.plot(years,turkeyPop)
plt.title('turkey population')
plt.show()


# In[27]:


import seaborn as sb


# In[28]:


smartphones=pd.read_csv('d://smartphones.csv')
smartphones


# In[41]:


sb.stripplot(x='OS',y='Capacity',data=smartphones,size=10)
plt.show()


# In[44]:


sb.swarmplot(x='OS',y='Capacity',data=smartphones,size=15,hue='Company')
plt.show()


# In[ ]:




