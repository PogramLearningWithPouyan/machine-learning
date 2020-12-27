#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing


# In[3]:


mydata=pd.read_csv('D://mydata.csv')


# In[4]:


mydata


# In[5]:


mydata.duplicated()


# In[6]:


mydata.drop_duplicates()


# In[7]:


mydata.drop_duplicates(['columns2'])


# In[8]:


my_source1=pd.read_csv('D://my_source1.csv')


# In[9]:


my_source2=pd.read_csv('D://my_source2.csv')


# In[10]:


my_source1


# In[11]:


my_source2


# In[28]:


my_concat=pd.concat([my_source1,my_source2],axis=0,ignore_index=True)


# In[30]:


my_concat


# In[31]:


my_concat.drop(['4'],axis=1,inplace=True)


# In[32]:


my_concat


# In[33]:


my_concat.drop_duplicates(inplace=True)


# In[34]:


my_concat


# In[35]:


smartphones=pd.read_csv('D://smartphones.csv')


# In[36]:


smartphones


# In[38]:


smartphones.describe()


# In[40]:


smartphones.OS.value_counts()


# In[41]:


smartphones.Company.value_counts()


# In[42]:


smartphones.Capacity.value_counts()


# In[46]:


cat_os=smartphones.groupby(smartphones['Company'])


# In[44]:


cat_os


# In[47]:


cat_os.mean()


# In[48]:


pd.crosstab(smartphones.OS,smartphones.Capacity)


# In[49]:


smartphones


# In[50]:


pd.pivot_table(smartphones,index='Name',columns='Company',values='Ram')


# In[51]:


smartphones.rename(index=smartphones.Name,inplace=True)


# In[52]:


smartphones


# In[54]:


smartphones.drop(['Name','Company'],axis=1,inplace=True)


# In[55]:


smartphones


# In[56]:


smartphones_data=pd.get_dummies(smartphones)


# In[57]:


smartphones_data


# In[58]:


smartphones_data.drop(['OS_windows'],axis=1,inplace=True)


# In[60]:


smartphones_data


# In[61]:


smartphones_data.describe()


# In[76]:


from sklearn.preprocessing import scale , normalize, minmax_scale


# In[64]:


scale_data=scale(smartphones_data)


# In[65]:


scale_data


# In[66]:


df_smartphones=pd.DataFrame(scale_data,index=smartphones_data.index,columns=smartphones_data.columns)


# In[67]:


df_smartphones


# In[74]:


norm_data=normalize(smartphones_data,norm='l1',axis=0)


# In[75]:


df_smartphones=pd.DataFrame(norm_data,index=smartphones_data.index,columns=smartphones_data.columns)
df_smartphones


# In[79]:


minmax_df=minmax_scale(smartphones_data,feature_range=(-1,1))


# In[80]:


df_smartphones=pd.DataFrame(minmax_df,index=smartphones_data.index,columns=smartphones_data.columns)
df_smartphones


# In[81]:


df=pd.DataFrame(np.array([1,2,3,4,10,27]))


# In[84]:


df.quantile(0.75)


# In[85]:


import matplotlib.pyplot as plt


# In[86]:


df.boxplot()


# In[ ]:




