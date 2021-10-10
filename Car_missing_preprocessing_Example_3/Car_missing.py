#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[8]:


# Predict the Price of Car


# In[3]:


dataset = pd.read_csv('cars_missing.csv')


# In[4]:


# Check Null Values
null_values = dataset.isnull().sum()
print('The Null Values in the Dataset is : \n{}'.format(null_values))


# In[6]:


# Display the Data
dataset.head(10)


# In[10]:


# Load the data into X & Y (numpy array)
X = dataset.iloc[:,1:].values
Y = dataset.iloc[:,0].values


# In[11]:


# Shape of X and Y
print('The Shape of X is : {}'.format(X.shape))
print('The Shape of Y is : {}'.format(Y.shape))


# # Handle Missing Values

# In[15]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean' )
X[:,[3,4]] = imputer.fit_transform(X[:,[3,4]])


# # Handle Categorical Data

# In[19]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X[:,2] = label_encoder.fit_transform(X[:,2])


# In[26]:


from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
dummy_v1 = onehot_encoder.fit_transform(X[:,[2]]).toarray()
X = np.hstack((X[:,[0,1]] , dummy_v1 , X[:,3:]))


# In[28]:


# After Apply OneHotEncoder the Shape of X is 
print('The Shape of X is : {}'.format(X.shape))


# # Normalization the Data [0-1]

# In[32]:


from sklearn.preprocessing import MinMaxScaler
normalize = MinMaxScaler()
X[:,[0,1,5,6,7,8,9,10]] = normalize.fit_transform(X[:,[0,1,5,6,7,8,9,10]])


# # Split the Data into Train & Test

# In[35]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size = 0.35 , random_state = 0)


# In[37]:


# Shape of Train and Test Data
print('The Shape of X Train is : {}'.format(x_train.shape))
print('The Shape of X Test is  : {}'.format(x_test.shape))
print('The Shape of Y Train is : {}'.format(y_train.shape))
print('The Shape of Y Test is  : {}'.format(y_test.shape))


# In[ ]:




