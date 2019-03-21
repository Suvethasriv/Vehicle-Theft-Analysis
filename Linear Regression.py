#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv("E:\Data.csv")
train.head()
train.info()
train.describe()
train.columns


# In[4]:


sns.pairplot(train)


# In[5]:


sns.distplot(train['Thefts'])


# In[57]:


train.corr()

