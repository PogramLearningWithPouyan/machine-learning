#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale


# In[20]:


boston= load_boston()
bo=scale(boston.data)
x=bo
y=boston.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[21]:


from sklearn.neural_network import MLPRegressor


# In[28]:


mlp=MLPRegressor(activation='identity',hidden_layer_sizes=(),
                 solver='sgd',alpha=0.0001,learning_rate_init=0.001,max_iter=500,early_stopping=True)
mlp.fit(x_train,y_train)
mlp.score(x_test,y_test)


# In[ ]:




