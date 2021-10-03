#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset = pd.read_csv('bike_buyers.csv')


# In[3]:


dataset.head(100)


# Check Null Values

# In[4]:


null_values = dataset.isnull().sum()
print('The Total Null Values in Dataset is : ')
print(null_values)


# Load the Data into X and Y (Variables)

# In[5]:


X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


# In[6]:


print('The Shape of The X is : {}'.format(X.shape))
print('The Shape of The Y is : {}'.format(Y.shape))


# Handle Missing Values

# In[7]:


# Nominal Data
from sklearn.impute import SimpleImputer
imputer_Nominal = SimpleImputer(missing_values = np.nan , strategy = 'most_frequent')
X[:,[1,2,7]] = imputer_Nominal.fit_transform(X[:,[1,2,7]])


# In[8]:


# Numeric Data
imputer_Numeric = SimpleImputer(missing_values = np.nan , strategy = 'mean')
X[:,[3,4,8,11]] = imputer_Numeric.fit_transform(X[:,[3,4,8,11]])


# Unique Values

# In[9]:


education = dataset['Education'].unique()
print('The Unique Values in the Education Attribute is : ')
print(education)


# In[10]:


occupation = dataset['Occupation'].unique()
print('The Unique Values in the Occupation Attribute is : ')
print(occupation)


# In[11]:


commute_distance = dataset['Commute Distance'].unique()
print('The Unique Values in the Commute Distance Attribute is : ')
print(commute_distance)


# In[12]:


region = dataset['Region'].unique()
print('The Unique Values in the Region Attribute is : ')
print(region)


# LabelEncoder and OneHotEncoder Apply (Nominal Data)

# In[13]:


# LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
X[:,1] = label_encoder.fit_transform(X[:,1])
X[:,2] = label_encoder.fit_transform(X[:,2])
X[:,7] = label_encoder.fit_transform(X[:,7])


# In[14]:


#OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
dummy_v1 = onehot_encoder.fit_transform(X[:,[5]]).toarray()
dummy_v2 = onehot_encoder.fit_transform(X[:,[6]]).toarray()
dummy_v3 = onehot_encoder.fit_transform(X[:,[9]]).toarray()
dummy_v4 = onehot_encoder.fit_transform(X[:,[10]]).toarray()
X = np.hstack((X[:,1:5],dummy_v1,dummy_v2,X[:,[7,8]],dummy_v3,dummy_v4,X[:,[11]] ))


# Convert the Range [0-1] (Numeric Data)

# In[15]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X[:,[2,3,15,24]] = mms.fit_transform(X[:,[2,3,15,24]])


# In[16]:


print('The Data in The X is : ')
print(X[0:3])


# Split The Data into (Traning and Testing)

# In[17]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test =  train_test_split(X,Y,test_size = 0.35,random_state =0)


# In[18]:


print('The Shape of X Train is : {}'.format(x_train.shape))
print('The Shape of X Test  is : {}'.format(x_test.shape))
print('The Shape of Y Train is : {}'.format(y_train.shape))
print('The Shape of Y Test  is : {}'.format(y_test.shape))


# In[ ]:




