#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Category Prediction
# In a set of documents, not only the words but the category of the words is also important;
# in which category of text a particular word falls. For example, we want to predict whether
# a given sentence belongs to the category email, news, sports, computer, etc. In the
# following example, we are going to use tf-idf to formulate a feature vector to find the
# category of documents. We will use the data from 20 newsgroup dataset of sklearn.


# In[ ]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


category_map = {'talk.religion.misc': 'Religion', 'rec.autos':
'Autos','rec.sport.hockey':'Hockey','sci.electronics':'Electronics',
'sci.space': 'Space'}


# In[ ]:


# Create the training set:
training_data = fetch_20newsgroups(subset='train',
categories=category_map.keys(), shuffle=True, random_state=5)


# In[ ]:


# Build a count vectorizer and extract the term counts:
vectorizer_count = CountVectorizer()
train_tc = vectorizer_count.fit_transform(training_data.data)
print("\nDimensions of training data:", train_tc.shape)
# print(train_tc)


# In[ ]:


# The tf-idf transformer is created as follows:
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)

print(train_tfidf)


# In[ ]:


# Now, define the test data:
input_data = [
'Discovery was a space shuttle',
'Hindu, Christian, Sikh all are religions',
'We must have to drive safely',
'Puck is a disk made of rubber',
'Television, Microwave, Refrigrated all uses electricity'
]


# In[ ]:


# The above data will help us train a Multinomial Naive Bayes classifier:
classifier = MultinomialNB().fit(train_tfidf, training_data.target)


# In[ ]:


# Transform the input data using the count vectorizer:
input_tc = vectorizer_count.transform(input_data)
# Now, we will transform the vectorized data using the tfidf transformer:
input_tfidf = tfidf.transform(input_tc)


# In[ ]:


# We will predict the output categories:
predictions = classifier.predict(input_tfidf)
# The output is generated as follows:
for sent, category in zip(input_data, predictions):
  print('\nInput Data:', sent, '\n Category:',         category_map[training_data.target_names[category]])


