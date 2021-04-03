#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[4]:


from sklearn.linear_model import LinearRegression


# In[5]:


x=np.arange(1,10)
y=np.array([28,25,26,31,32,29,30,35,36])


# In[6]:


plt.scatter(x,y)
plt.show()


# In[7]:


x=x.reshape(-1,1)
y=y.reshape(-1,1)
x


# In[8]:


y


# In[9]:


reg=LinearRegression()
reg.fit(x,y)


# In[10]:


y_predict=reg.predict(x)


# In[ ]:





# In[12]:


plt.scatter(x,y)
plt.plot(x,y_predict)
plt.show()


# In[13]:


from sklearn.datasets import load_boston


# In[14]:


boston=load_boston()
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)
boston_df


# In[15]:


boston_df['price']=boston.target
boston_df


# In[16]:


x=boston.data
y=boston.target


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,
                                              random_state=42)


# In[20]:


reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)


# In[21]:


plt.scatter(y_test,y_pred)
plt.xlabel('price')
plt.ylabel('predictet price')
plt.show()


# In[22]:


from sklearn.metrics import mean_squared_error


# In[25]:


mse=mean_squared_error(y_test,y_pred)
mse


# In[26]:


new_x=boston.data[:,[1,2]]
new_y=boston.target
new_x_trian,new_x_test,new_y_train,new_y_test=train_test_split(new_x,new_y,test_size=0.3, random_state=42)
new_reg=LinearRegression()
new_reg.fit(new_x_trian,new_y_train)
new_y_pred=new_reg.predict(new_x_test)
new_mse=mean_squared_error(new_y_test,new_y_pred)
new_mse


# In[27]:


from sklearn.model_selection import cross_val_score


# In[28]:


reg=LinearRegression()
cv_score=cross_val_score(reg,boston.data,boston.target,cv=5)
cv_score


# In[30]:


np.mean(cv_score)


# In[31]:


from sklearn.linear_model import Lasso


# In[37]:


lasso =Lasso(alpha=0.1,normalize=True)
lasso.fit(boston.data,boston.target)
lasso_coef=lasso.coef_
lasso_coef


# In[39]:


plt.plot(range(13),lasso_coef)
plt.xticks(range(13), boston.feature_names)
plt.ylabel=('coefficents')
plt.show()


# In[41]:


lasso =Lasso(alpha=0.1,normalize=True)
lasso.fit(x_train,y_train)
y_lasso=lasso.predict(x_test)
lasso_mse=mean_squared_error(y_test,y_lasso)
lasso_mse


# In[43]:


from sklearn.linear_model import Ridge


# In[44]:


ridge=Ridge(alpha=0.1,normalize=True)
ridge.fit(x_train,y_train)
y_ridge=ridge.predict(x_test)
ridge_mse=mean_squared_error(y_test,y_ridge)
ridge_mse


# In[ ]:




