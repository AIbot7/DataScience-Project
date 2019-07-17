#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;
sns.set()
data=pd.read_csv('heart.csv')


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Import tools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,roc_curve


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.describe()


# In[ ]:


plt.figure(figsize=(18,10))
sns.heatmap(data.corr(), annot = True, cmap='cool')
plt.show()


# In[ ]:


sns.countplot(data.target, palette=['green', 'red'])
plt.title("[0] == Not Disease, [1] == Disease")


# In[ ]:


plt.figure(figsize=(18, 10))
sns.countplot(x='age', hue='target', data=data, palette=['#1CA53B', 'red'])
plt.legend(["Haven't Disease", "Have Disease"])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


fig, axes = plt.subplots(3, 2, figsize=(12,12))
fs = ['cp', 'fbs', 'restecg','exang', 'slope', 'ca']
for i, axi in enumerate(axes.flat):
    sns.countplot(x=fs[i], hue='target', data=data, palette='bwr', ax=axi) 
    axi.set(ylabel='Frequency')


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='trestbps',y='thalach',data=data,hue='target')
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='chol',y='thalach',data=data,hue='target')
plt.show()


# In[ ]:


plt.scatter(x=data.age[data.target==1], y=data.thalach[(data.target==1)], c="red")
plt.scatter(x=data.age[data.target==0], y=data.thalach[(data.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# In[ ]:


# Define our feasures and leabels
X = np.array(data.drop('target', 1))
y = np.array(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Let's create our model and classifer
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[ ]:


print(y_pred)


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
acc = round(accuracy, 4) * 100
print('Model accuracy is {} %'.format(acc))


# In[ ]:


mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cmap='rainbow')
plt.xlabel('predicted value')
plt.ylabel('true value');
plt.show();


# In[ ]:


print(classification_report(y_test, y_pred, target_names=['Non Disease', 'Disease']))


# In[ ]:


clf.score(X_test, y_test)


