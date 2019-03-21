#!/usr/bin/env python
# coding: utf-8

# In[31]:


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


# In[32]:


train = pd.read_csv("E:\Data.csv")


# In[33]:


print("***** Train_Set *****")
print(train.head())
print("\n")


# In[34]:


print("***** Train_Set *****")
print(train.describe())
print("\n")


# In[35]:


print(train.columns.values)


# In[36]:


train.isna().head()


# In[39]:


print("*****In the train set*****")
print(train.isna().sum())
print("\n")


# In[40]:


train = train.applymap(lambda Charge_Type: 1 if Charge_Type == True else Charge_Type)
train = train.applymap(lambda Charge_Type: 0 if Charge_Type == False else Charge_Type)


# In[41]:


train.fillna(train.mean(), inplace=True)


# In[42]:


print(train.isna().sum())


# In[43]:


train[["State", "Thefts"]].groupby(['State'], as_index=False).mean().sort_values(by='Thefts', ascending=False)


# In[44]:


train[["State", "Rank"]].groupby(['State'], as_index=False).mean().sort_values(by='Rank', ascending=False)


# In[45]:


train[["Make/Model", "Thefts"]].groupby(['Make/Model'], as_index=False).mean().sort_values(by='Thefts', ascending=False)


# In[46]:


train[["Make/Model", "Rank"]].groupby(['Make/Model'], as_index=False).mean().sort_values(by='Rank', ascending=False)


# In[47]:


train.info()


# In[48]:


y = np.array(train['State'])


# In[49]:


train.info()


# In[50]:


kmeans = KMeans(n_clusters=2)


# In[51]:


correct = 0


# In[52]:


data =  pd.read_csv("E:\Data.csv")
cluster_X = data.iloc[:,1:]


# In[53]:


K_Means = KMeans(3)


# In[54]:


prediction_dataset = data.copy()
prediction_dataset['cluters'] = 10000


# In[30]:


plot.scatter(prediction_dataset['Rank'],prediction_dataset['Thefts'],c=prediction_dataset['cluters'],cmap='rainbow')
plot.xlim(0,11)
plot.ylim(0,2000)
plot.show()


# In[ ]:




