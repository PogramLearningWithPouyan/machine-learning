#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing


# In[84]:


country=pd.read_csv('D://c_data.csv',header=1)


# In[85]:


country


# In[86]:


country.rename(columns={'CountryName':'Name','CountryCode':'Code',
                        'Population growth':'pop_growth','Total population':'pop'
                       ,'Area (sq. km)':'Area'},inplace=True)


# In[87]:


country


# In[88]:


country.drop('Code',axis=1,inplace=True)


# In[89]:


country


# In[90]:


country.drop('1',axis=1,inplace=True)


# In[91]:


country


# In[92]:


country.rename(index=country.Name,inplace=True)


# In[93]:


country.drop('Name',axis=1,inplace=True)


# In[94]:


country


# In[95]:


country.info()


# In[96]:


country.describe()


# In[97]:


maxpop=country['pop'].max()


# In[98]:


country['pop'][country['pop']==maxpop]


# In[99]:


country.drop('World',axis=0, inplace=True)


# In[100]:


country


# In[101]:


country.isnull()


# In[102]:


country.info()


# In[103]:


country.replace('?',np.nan,inplace=True)


# In[104]:


country


# In[105]:


country.isnull()


# In[106]:


country.isnull().sum()


# In[107]:


country.dropna(axis=0)


# In[108]:


country.fillna(0)


# In[111]:


country


# In[112]:


country.fillna({'pop_growth':0,'pop':100000000,'Area':500000})


# In[113]:


country.fillna(method='ffill')


# In[117]:


from sklearn.impute import SimpleImputer


# In[124]:


i=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)


# In[125]:


i.fit(country)


# In[126]:


new_df=i.transform(country)


# In[127]:


new_df


# In[ ]:




