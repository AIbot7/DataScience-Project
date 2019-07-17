#!/usr/bin/env python
# coding: utf-8


# 
# Objective is to recommend books to a user based on purchase history and behavior of other users
# 
# ## Getting Started
# 
# Import some libraries you will need:

# In[92]:


import numpy as np
import pandas as pd


# Read the data using pandas in DataFrame df

# In[93]:


df_user = pd.read_csv('BX-Users.csv',encoding='latin-1',sep=';')



# Take a quick look at the user data.

# In[94]:


df_user.head()


# Clean up NaN values

# Read the books Data and explore

# In[120]:


#column_names = ['isbn', 'book_title']
#df_books = pd.read_csv('BX-Books.csv', encoding='latin-1',sep=';')
df_books = pd.read_csv('BX-Books.csv',encoding='latin-1',sep=';',error_bad_lines=False)

# In[121]:


df_books.head()



# In[122]:
df_books.shape

#%%


df = pd.read_csv('BX-Book-Ratings.csv',encoding='latin-1',nrows=10000)


# In[123]:


df.head()


# In[124]:


df.describe()


# Merge the dataframes. For all practical purposes, User Master Data is not required. So, ignore dataframe df_user

# In[125]:


df = pd.merge(df,df_books,on='isbn')
df.head()


# Now, let's take a quick look at the number of unique users and books.

# In[126]:


n_users = df.user_id.nunique()
n_books = df.isbn.nunique()

print('Num. of Users: '+ str(n_users))
print('Num of Books: '+str(n_books))


# Convert ISBN to numeric numbers in order 

# In[127]:


isbn_list = df.isbn.unique()
print(" Length of isbn List:", len(isbn_list))
def get_isbn_numeric_id(isbn):
    #print ("  isbn is:" , isbn)
    itemindex = np.where(isbn_list==isbn)
    return itemindex[0][0]


# Do the same for user_id , convert it into numeric and in order

# In[128]:


userid_list = df.user_id.unique()
print(" Length of user_id List:", len(userid_list))
def get_user_id_numeric_id(user_id):
    #print ("  isbn is:" , isbn)
    itemindex = np.where(userid_list==user_id)
    return itemindex[0][0]


# Converting both user_id and isbn to ordered list i.e. from 0...n-1

# In[129]:


df['user_id_order'] = df['user_id'].apply(get_user_id_numeric_id)


# In[130]:


df['isbn_id'] = df['isbn'].apply(get_isbn_numeric_id)
df.head()


# Re-index columns to build matrix later on

# In[133]:


new_col_order = ['user_id_order', 'isbn_id', 'rating', 'book_title', 'book_author','year_of_publication','publisher','isbn','user_id']
df = df.reindex(columns= new_col_order)
df.head()


# ## Train Test Split
# 
# In[134]:


from sklearn.cross_validation import train_test_split
train_data, test_data = train_test_split(df, test_size=0.30)


# ## Approach: will Use Memory-Based Collaborative Filtering
# 

# In[135]:


#Create two user-book matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_books))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  

test_data_matrix = np.zeros((n_users, n_books))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]



# In[136]:


from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')


# In[137]:


user_similarity


# Next step is to make predictions

# In[138]:


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred


# In[139]:


item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')



# In[140]:


from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


# In[141]:


print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))


# ### Both the approach yield almost same result

# # End

