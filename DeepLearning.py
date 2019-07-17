#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Project: Pet Classifier using CNN
# 

# 
# Data Set
# - A production grade program as 10,000 training images
# - This is a small program with 20 images of cats and 20 images of dogs. 
# - The evaluation set has 10 images of cats and 10 images of dogs
# 


# ### Import modules

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import sys

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# ### Set hyper parameters
# - Run the program with three num_steps : 100,200,300

# In[2]:


reset_graph()

img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
img_shape = (img_size, img_size)
trainpath='./data/train'
testpath='./data/test'
labels = {'cats': 0, 'dogs': 1}
fc_size=32 #size of the output of final FC layer
num_steps=300 #Try 100, 200, 300. number of steps that training data should be looped. Usually 20K
tf.logging.set_verbosity(tf.logging.INFO)


# ### Read the image dataset

# In[3]:


def read_images_classes(basepath,imgSize=img_size):
    image_stack = []
    label_stack = []

    for counter, l in enumerate(labels):
        path = os.path.join(basepath, l,'*g')
        for img in glob.glob(path):
            one_hot_vector =np.zeros(len(labels),dtype=np.int16)
            one_hot_vector[counter]=1
            image = cv2.imread(img)
            im_resize = cv2.resize(image,img_shape, interpolation=cv2.INTER_CUBIC)
            image_stack.append(im_resize)
            label_stack.append(labels[l])            
    return np.array(image_stack), np.array(label_stack)

X_train, y_train=read_images_classes(trainpath)
X_test, y_test=read_images_classes(testpath)

#test a sample image
print('length of train image set',len(X_train))
print('X_data shape:', X_train.shape)
print('y_data shape:', y_train.shape)

fig1 = plt.figure() 
ax1 = fig1.add_subplot(2,2,1) 
img = cv2.resize(X_train[0],(64,64), interpolation=cv2.INTER_CUBIC)
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(y_train[0])
plt.show()



# 
# The model should have the following layers
# - input later
# - conv layer 1 with 32 filters of kernel  size[5,5],
# - pooling layer 1 with pool size[2,2] and stride 2
# - conv layer 2 with 64 filters of kernel  size[5,5],
# - pooling layer 2 with pool size[2,2] and stride 2
# - dense layer whose output size is fixed in the hyper parameter: fc_size=32
# - drop out layer with droput probability 0.4
# - predict the class by doing a softmax on the output of the dropout layers
# 
# Training
# - For training fefine the loss function and minimize it
# - For evaluation calculate the accuracy
# 
# Reading Material
# - For ideas look at tensorflow layers tutorial

# ### The cnn_model_fn has to be defined here by the student

# In[4]:


def cnn_model_fn(features, labels, mode):

    ...
    ...


# ### Run the tensorflow model
# 
# This section will use the model defined by the student and run the training and evaluation step

# In[5]:


#X_train = np.array((X_train/255.0),dtype=np.float16)
#X_test = np.array((X_test/255.0), dtype=np.float16)
X_train = np.array((X_train/255.0),dtype=np.float32)
X_test = np.array((X_test/255.0), dtype=np.float32)

pets_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/pets_convnet_model")
#pets_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn)
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X_train}, y=y_train, batch_size=10,
                                                      num_epochs=None, shuffle=True)
pets_classifier.train(input_fn=train_input_fn, steps=num_steps, hooks=[logging_hook])
eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X_test}, y=y_test, num_epochs=1,shuffle=False)
eval_results = pets_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)


# In[ ]:





