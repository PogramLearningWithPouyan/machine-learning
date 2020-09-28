#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


a=np.array([[1,2],[3,4]])
a


# In[4]:


b=np.matrix([[1,2],[3,4]])
b


# In[5]:


a@a


# In[6]:


np.dot(a,a)


# In[7]:


np.multiply(a,a
           )


# In[8]:


np.prod(a)


# In[2]:


my_array=np.array([1,2,3])
my_array+5


# In[3]:


c=np.ones((3,3))
c


# In[4]:


d=np.array([5,6,7])
c+d


# In[5]:


g=np.ones((3,1))
g


# In[6]:


n=np.array([5,6,7])
g+n


# In[6]:


np.sum(a)


# In[9]:


np.cumsum(a , axis=1)


# In[10]:


np.subtract(a,a)


# In[11]:


np.divide([5,6,7],3)


# In[12]:


np.floor_divide([5,6,7],3)


# In[13]:


np.math.sqrt(5)


# In[15]:


np.math.nan


# In[16]:


np.math.inf


# In[17]:


np.random.uniform(1,5,(2,3))


# In[18]:


np.random.standard_normal((2,1))


# In[3]:


np.arange(1,10,3)


# In[4]:


np.linspace(1,10,4)


# In[5]:


my_mask=a>2
a[my_mask]


# In[6]:


my_mask2=np.logical_(a>1,a<4)


# In[7]:


a[my_mask2]


# In[8]:


np.ones((1,3))


# In[9]:


np.zeros((2,3))


# In[10]:


a


# In[11]:


np.size(a)


# In[12]:


np.shape(a)


# In[13]:


aa=np.array([1,2,3,4,1,3])
aa


# In[14]:


bb=([7,8,9,1,3])


# In[15]:


bb


# In[16]:


np.unique(a)


# In[17]:


np.union1d(aa,bb)


# In[18]:


np.intersect1d(aa,bb)


# In[19]:


np.mean(aa)


# In[20]:


np.median(aa)


# In[21]:


np.std(aa)


# In[22]:


np.var(aa)


# In[24]:


chandJomleii=np.array([1,2,2])


# In[26]:


np.polyval(chandJomleii,2)


# In[27]:


np.polyder(chandJomleii)


# In[28]:


np.polyint(chandJomleii)


# In[ ]:




