#!/usr/bin/env python
# coding: utf-8

# ## Importing Packages

# In[21]:


# Load Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# ## Retrieve Data from Employee Data Set

# In[22]:


# Read Data
data= pd.read_csv('INX_Future_Inc_Employee_Performance')


# In[17]:


data.head()


# In[19]:


data.dtypes.index


# ## Drop Employee Number from the Table

# In[23]:


#Drop the column
data=data.drop('EmpNumber', axis=1)
data.head()


# ## Type Conversion of data values from String to Float using bulit-in function (Data Manipulation)

# In[24]:


inx1=pd.get_dummies(data,drop_first=True)
inx1.head()


# In[ ]:





# In[25]:


inx1.columns


# ## Fetching the Column Values from Table for Reading Variables except PerformanceRating  Column

# In[26]:


x=inx1.drop('PerformanceRating',axis=1)


# In[27]:


y=inx1.loc[:,['PerformanceRating']]


# ## Split Train Test

# In[35]:


x_train,x_test,y_train,y_test =train_test_split(x,y, test_size = 0.3, random_state = 20)


# ## Implementatioon of Random Forest Model

# In[29]:


rf_model =  RandomForestClassifier(random_state = 10)
rf_model.fit(x_train,y_train)


# ## Predicting Accuracy

# In[36]:


y_pred=rf_model.predict(x_test)


# In[37]:


print(accuracy_score(y_test,y_pred))


# In[ ]:





