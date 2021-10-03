#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset = pd.read_csv('Data_Set_1.csv')


# In[3]:


dataset.head(10)


# Check Null Values

# In[5]:


null_values = dataset.isnull().sum()
print('The Total Null Values is : ')
print(null_values)


# Load The Data into X and Y (Variables)

# In[6]:


X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


# In[7]:


print('The Shape of X is : {}'.format(X.shape))
print('The Shape of Y is : {}'.format(Y.shape))


# Handle Missing values (Numeric Data)

# In[10]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
X[:,[1,2]] = imputer.fit_transform(X[:,[1,2]])


# LabelEncoder and OneHotEncoder Apply (Nominal Data)

# In[17]:


# LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X[:,0] = label_encoder.fit_transform(X[:,0])
Y = label_encoder.fit_transform(Y)


# In[30]:


# OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
dummy_v1 = onehot_encoder.fit_transform(X[:,[0]]).toarray()
X = np.hstack((dummy_v1 , X[:,[1,2]]))


# In[33]:


print('The Shape of X is : {}'.format(X.shape))


# Convert the Range [0-1] (Numeric Data)

# In[36]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X[:,[3,4]] = mms.fit_transform(X[:,[3,4]])


# In[40]:


print('The Data in the X is : ')
print(X[:3])


# Split the data into (Training and Testing)

# In[42]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = 0.3 , random_state = 0)


# In[43]:


print('The Shape of X Train is : {}'.format(x_train.shape))
print('The Shape of X Test  is : {}'.format(x_test.shape))
print('The Shape of Y Train is : {}'.format(y_train.shape))
print('The Shape of Y Test  is : {}'.format(y_test.shape))

