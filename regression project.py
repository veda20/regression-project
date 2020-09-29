#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing databases
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


# In[2]:


# understanding the data sets
boston = load_boston()
print (boston.DESCR)


# In[3]:


# access data attributes
dataset = boston.data
for name , index in enumerate(boston.feature_names):
     print(index,name)


# In[4]:


# reshaping data 
data = dataset [:,12].reshape(-1,1)


# In[5]:


# shape of the data 
np.shape(dataset)


# In[6]:


# target value 
target = boston.target.reshape(-1,1)


# In[7]:


# shape of target 
np.shape(target)


# In[11]:


# ensure matplotlib is working 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color = "green")
plt.xlabel("lower income population")
plt.ylabel("cost of house")
plt.show()


# In[13]:


# regression
from sklearn.linear_model import LinearRegression

#creating a regression model
reg = LinearRegression()

#fit the model
reg.fit(data, target)


# In[14]:


# prediction 
pred = reg.predict(data)


# In[16]:


# ensure matplotlib is working 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color = "red")
plt.plot(data,pred,color = "green")
plt.xlabel("lower income population")
plt.ylabel("cost of house")
plt.show()


# In[17]:


# circumventing curve issue using polynomial regression model
from sklearn.preprocessing import PolynomialFeatures

# to allow merging of models 
from sklearn.pipeline import make_pipeline


# In[18]:


model =  make_pipeline(PolynomialFeatures(3) , reg)


# In[19]:


model.fit(data,target)


# In[20]:


pred = model.predict(data)


# In[21]:


# ensure matplotlib is working 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color = "red")
plt.plot(data,pred,color = "green")
plt.xlabel("lower income population")
plt.ylabel("cost of house")
plt.show()


# In[22]:


# r_2 metric
from sklearn.metrics import r2_score


# In[24]:


# predict
r2_score(pred,target)


# In[ ]:




