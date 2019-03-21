#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt


# In[8]:


print(df.head())
df.info()


# In[4]:


df['Make/Model']


# In[5]:


import matplotlib.pyplot as plt
import pandas as pd
file = r'E:\Pro\Alabama.xlsx'
df = pd.read_excel(file)
State = df["Make/Model"]
data = df["Thefts"]
colors = ['yellow', 'blue','red','green','violet', 'lightblue', 'pink', 'lightgreen','black','brown'] 
plt.pie(data, labels=State, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Alabama")
plt.show()


# In[6]:


import matplotlib.pyplot as plt
import pandas as pd
file = r'E:\Pro\Alaska.xlsx'
df = pd.read_excel(file)
State = df["Make/Model"]
data = df["Thefts"]
colors = ['yellow','blue','red','green','violet','lightblue', 'purple', 'lightgreen','pink','brown'] 
plt.pie(data, labels=State, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=180)
plt.title("Alaska")
plt.show()


# In[ ]:




