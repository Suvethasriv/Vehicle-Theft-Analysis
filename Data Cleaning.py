#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


train = pd.read_csv('E:\Data.csv')


# In[3]:


train.head()


# In[4]:


print(train.columns.values)


# In[5]:


train.isna().head()


# In[6]:


print("*****In the train set*****")
print(train.isna().sum())
print("\n")


# In[7]:


train.fillna(train.mean(), inplace=True)


# In[8]:


print(train.isna().sum())


# In[9]:


train['State'].head()


# In[10]:


train['Thefts'].head()


# In[11]:


train[['State', 'Rank']].groupby(['State'], as_index=False).mean().sort_values(by='Rank', ascending=False)


# In[14]:


train.corr()


# In[12]:


train.apply(lambda x: x.count(), axis=1)


# In[13]:


train.isnull()


# In[14]:


missing_values = ["n/a", "na", "--"]
train = pd.read_csv("E:\Data.csv", na_values = missing_values)


# In[15]:


train.isnull()


# In[19]:


print (train.isnull().sum())


# In[20]:


print (train.isnull().values.any())


# In[21]:


print (train.isnull().sum().sum())


# In[22]:


null_columns=train.columns[train.isnull().any()] 


# In[30]:


print(train[train["Make/Model"].isnull()][null_columns])


# In[31]:


print (train.isnull().sum())


# In[ ]:




