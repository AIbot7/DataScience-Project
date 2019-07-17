#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("run_or_walk.csv")


# In[4]:


df.info()


# In[5]:


df.columns


# In[6]:


from sklearn.model_selection import train_test_split
X, y = df.iloc[:, 5:].values,df.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[7]:


print(X_train.shape)
print(y_test[0:10])


# In[8]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()


# In[9]:


classifier.fit(X_train,y_train)


# In[10]:


y_predict = classifier.predict(X_test)


# In[11]:


from sklearn.metrics import accuracy_score


# In[12]:


accuracy = accuracy_score(y_predict,y_test)


# In[13]:


print(accuracy)


# In[14]:


from sklearn.metrics import confusion_matrix


# In[15]:


conf_mat =confusion_matrix(y_predict,y_test)


# In[16]:


print(conf_mat)


# In[17]:


from sklearn.metrics import classification_report
target_names = ["Walk","Run"]


# In[18]:


print(classification_report(y_test, y_predict, target_names=target_names))


# In[19]:


df.info()


# In[23]:


from sklearn.model_selection import train_test_split
X, y = df.iloc[:, [5,6,7]].values,df.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[24]:


classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)
accuracy_score(y_predict,y_test)


# In[25]:


print(conf_mat)


# In[ ]:





# In[ ]:





