#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv("E:\Data.csv")


# In[3]:


print("***** Train_Set *****")
print(train.head())
print("\n")


# In[4]:


print("***** Train_Set *****")
print(train.describe())
print("\n")


# In[5]:


print(train.columns.values)


# In[6]:


train.isna().head()


# In[7]:


print("*****In the train set*****")
print(train.isna().sum())
print("\n")


# In[8]:


train = train.applymap(lambda Thefts: 1 if Thefts == True else Thefts)
train = train.applymap(lambda Thefts: 0 if Thefts == False else Thefts)


# In[9]:


train.fillna(train.mean(), inplace=True)


# In[10]:


print(train.isna().sum())


# In[11]:


train[["State", "Thefts"]].groupby(['State'], as_index=False).mean().sort_values(by='Thefts', ascending=False)


# In[12]:


train[["State", "Rank"]].groupby(['State'], as_index=False).mean().sort_values(by='Rank', ascending=False)


# In[13]:


train[["Make/Model", "Thefts"]].groupby(['Make/Model'], as_index=False).mean().sort_values(by='Thefts', ascending=False)


# In[14]:


train[["Make/Model", "Rank"]].groupby(['Make/Model'], as_index=False).mean().sort_values(by='Rank', ascending=False)


# In[15]:


train.info()


# In[16]:


y = np.array(train['State'])


# In[17]:


train.info()


# In[18]:


kmeans = KMeans(n_clusters=2)


# In[19]:


correct = 0


# In[20]:


data =  pd.read_csv("E:\Data.csv")
cluster_X = data.iloc[:,1:]


# In[21]:


K_Means = KMeans(3)


# In[22]:


prediction_dataset = data.copy()
prediction_dataset['cluters'] = 10000


# In[23]:


plot.scatter(prediction_dataset['Rank'],prediction_dataset['Thefts'],c=prediction_dataset['cluters'],cmap='rainbow')
plot.xlim(0,11)
plot.ylim(0,2000)
plot.show()


# In[ ]:




