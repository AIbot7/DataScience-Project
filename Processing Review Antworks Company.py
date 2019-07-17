#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import required libraries
import pandas as pd


# In[ ]:


get_ipython().system('pip install patool')


# In[ ]:


import patoolib
patoolib.extract_archive("antworks_assignment.zip", outdir=".")


# In[ ]:


#get the sentiment dataset
df_sentiment = pd.read_csv('train.csv',sep='~')
test = pd.read_csv('test.csv',sep='~')


# In[ ]:


#view first 10 observations. 
# 1 indicates positive review and 0 indicate negative review
df_sentiment.head(10)


# In[ ]:


test.head()


# In[ ]:


df_sentiment.shape


# In[ ]:


# view more information about the review data using describe method
df_sentiment.describe()


# In[ ]:


#view more info on data
df_sentiment.info()


# In[ ]:


# view data using group by and describe method
df_sentiment.groupby('Is_Response').describe()


# In[ ]:


# Verify length of the messages and also add it also as a new column (feature)
df_sentiment['length'] =df_sentiment['Description'].apply(len)


# In[ ]:


# view first 5 messages with length

df_sentiment.head()


# In[ ]:


#view first 
df_sentiment[df_sentiment['length']>50]['Description'].iloc[0]


# In[ ]:


# start text processing with vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer#CountVectorizer
#vectorizer = CountVectorizer()
vectorizer =TfidfVectorizer()


# 

# In[ ]:


# define a function to get rid of stopwords present in the messages
def message_text_process(mess):
    # Check characters to see if there are punctuations
    no_punctuation = [char for char in mess if char not in string.punctuation]
    # now form the sentence.
    no_punctuation = ''.join(no_punctuation)
    # Now eliminate any stopwords
    return [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]


# In[ ]:


import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[ ]:


# bag of words by applying the function and fit the data (comment) into it

# bag_of_words = CountVectorizer(analyzer=message_text_process).fit(df_sentiment['Description'])
bag_of_words = TfidfVectorizer(analyzer=message_text_process).fit(df_sentiment['Description'])
# test = TfidfVectorizer(analyzer=message_text_process).fit(test['Description'])


# In[ ]:


bag_of_words


# In[ ]:


# apply transform method for the bag of words
comment_bagofwords = bag_of_words.transform(df_sentiment['Description'])
comment_bagofwords_test = test.transform(test['Description'])


# In[ ]:


comment_bagofwords


# In[ ]:


# apply tfidf transformer and fit the bag of words into it (transformed version)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(comment_bagofwords)


# In[ ]:


# print shape of the tfidf 
comment_tfidf = tfidf_transformer.transform(comment_bagofwords)
print (comment_tfidf.shape)


# In[ ]:


print (comment_tfidf)


# In[ ]:


comment_tfidf


# In[ ]:



from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)


# In[ ]:


#choose naive Bayes model to detect the Good or bad and fit the tfidf data into it
from sklearn.naive_bayes import MultinomialNB
sentiment_detection_model = MultinomialNB().fit(comment_tfidf,df_sentiment['Is_Response'])


# In[ ]:





# In[ ]:


# check model for the predicted  and expected value say for comment# 1 and comment#5
comment = df_sentiment['Is_Response']
bag_of_words_for_comment = bag_of_words.transform([comment])
tfidf = tfidf_transformer.transform(bag_of_words_for_comment)


# In[ ]:


y_pred=sentiment_detection_model.predict(tfidf)
y_pred


# In[ ]:


print ('predicted review ', sentiment_detection_model.predict(tfidf)[0])
print ('expected review ', df_sentiment['Is_Response'][0]) # label[4])


# In[ ]:


# print ('predicted review ', sentiment_detection_model.predict(tfidf))
# print ('expected review ', df_sentiment['Is_Response']) # label[4])


# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score(tfidf, y_pred))


# In[ ]:


y_pred=sentiment_detection_model.predict(tfidf)
y_pred


# In[ ]:


comment


# In[ ]:


comment=comment.map(dict(Good=1, Bad=0))

comment


# In[ ]:





# In[ ]:





