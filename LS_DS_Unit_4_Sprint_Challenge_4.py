#!/usr/bin/env python
# coding: utf-8

# <img align="left" src="https://lever-client-logos.s3.amazonaws.com/864372b1-534c-480e-acd5-9711f850815c-1524247202159.png" width=200>
# <br></br>
# <br></br>
# 
# # Major Neural Network Architectures Challenge
# ## *Data Science Unit 4 Sprint 3 Challenge*
# 
# In this sprint challenge, you'll explore some of the cutting edge of Data Science. This week we studied several famous neural network architectures: 
# recurrent neural networks (RNNs), long short-term memory (LSTMs), convolutional neural networks (CNNs), and Generative Adverserial Networks (GANs). In this sprint challenge, you will revisit these models. Remember, we are testing your knowledge of these architectures not your ability to fit a model with high accuracy. 
# 
# __*Caution:*__  these approaches can be pretty heavy computationally. All problems were designed so that you should be able to achieve results within at most 5-10 minutes of runtime on Colab or a comparable environment. If something is running longer, doublecheck your approach!
# 
# ## Challenge Objectives
# *You should be able to:*
# * <a href="#p1">Part 1</a>: Train a RNN classification model
# * <a href="#p2">Part 2</a>: Utilize a pre-trained CNN for objective detection
# * <a href="#p3">Part 3</a>: Describe the difference between a discriminator and generator in a GAN
# * <a href="#p4">Part 4</a>: Describe yourself as a Data Science and elucidate your vision of AI

# <a id="p1"></a>
# ## Part 1 - RNNs
# 
# Use an RNN to fit a multi-class classification model on reuters news articles to distinguish topics of articles. The data is already encoded properly for use in an RNN model. 
# 
# Your Tasks: 
# - Use Keras to fit a predictive model, classifying news articles into topics. 
# - Report your overall score and accuracy
# 
# For reference, the [Keras IMDB sentiment classification example](https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py) will be useful, as well the RNN code we used in class.
# 
# __*Note:*__  Focus on getting a running model, not on maxing accuracy with extreme data size or epoch numbers. Only revisit and push accuracy if you get everything else done!

# In[20]:


from tensorflow.keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=723812,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)


# In[21]:


# Demo of encoding

# we got the indices before now we get the word index from reuters word index.json

word_index = reuters.get_word_index(path="reuters_word_index.json")

print(f"Iran is encoded as {word_index['iran']} in the data")
print(f"London is encoded as {word_index['london']} in the data")
print("Words are encoded as numbers in our dataset.")


# In[22]:


print('# of training samples: {}'.format(len(x_train)))
print('# of text samples: {}'.format(len(x_test)))

num_classes = max(y_train) + 1
print('# of classes: {}'.format(num_classes))


# In[23]:


print(x_train[0])
print(y_train[0])


# In[24]:


# let's see which words are the most common:
print('pak:',word_index['pakistan'])

#print('us:', word_index['united states'])

print('peace:', word_index['peace'])

print('computer: ', word_index['computer'])

print('const:', word_index['constitution'])


# In[27]:


from keras.preprocessing.text import Tokenizer

max_words = 10000
tokenizer = Tokenizer(num_words =max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')


# In[28]:


# let's do the same with y_test
import keras 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[29]:


print(x_train.shape)
print(x_train[0])

print(y_train.shape)
print(y_train[0])


# In[35]:


from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation, Dropout
from keras.layers import LSTM


print('Build model...')

model = Sequential()
model.add(Dense(512, input_shape=(max_words, )))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

batch_size = 64
epochs = 20

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

score = model.evaluate(x_test,y_test,batch_size=batch_size, verbose=1)

print('test loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))


# Conclusion - RNN runs, and gives pretty decent improvement over a naive model. To *really* improve the model, more playing with parameters would help. Also - RNN may well not be the best approach here, but it is at least a valid one.

# <a id="p2"></a>
# ## Part 2- CNNs
# 
# ### Find the Frog
# 
# Time to play "find the frog!" Use Keras and ResNet50 (pre-trained) to detect which of the following images contain frogs:
# 
# <img align="left" src="https://d3i6fh83elv35t.cloudfront.net/newshour/app/uploads/2017/03/GettyImages-654745934-1024x687.jpg" width=400>
# 

# In[36]:


get_ipython().system('pip install google_images_download')


# In[42]:


from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()
arguments = {"keywords": "animal pond", "limit": 50, "print_urls": True}
absolute_image_paths = response.download(arguments)


# At time of writing at least a few do, but since the Internet changes - it is possible your 5 won't. You can easily verify yourself, and (once you have working code) increase the number of images you pull to be more sure of getting a frog. Your goal is to validly run ResNet50 on the input images - don't worry about tuning or improving the model.
# 
# *Hint* - ResNet 50 doesn't just return "frog". The three labels it has for frogs are: `bullfrog, tree frog, tailed frog`
# 
# *Stretch goal* - also check for fish.

# In[43]:


# TODO - your code!

import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from IPython.display import Image

def process_img_path(img_path):
  return image.load_img(img_path, target_size=(224, 224))

def img_list_predict(img_list):
  model = ResNet50(weights='imagenet')
  for img_url in img_list:
    display(Image(filename=img_url, width=600))
    img = process_img_path(img_url)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    results = decode_predictions(features, top=3)[0]
    print(results)
  return

images = absolute_image_paths[0]['animal pond']
img_list_predict(images)


# <a id="p3"></a>
# ## Part 3 - Generative Adverserial Networks (GANS)
# 
# Describe the difference between a discriminator and generator in a GAN in your own words.
# 
# __*Your Answer:*__ 

# #### Generator ####
# A generator creates an image from noise and tries to fool the discriminator in thinking that it is a real image.
# 
# #### Discriminator ####
# The job of a discriminator is to look at the image and be able to tell it from real to fake
# 
# How GAN works is the generator creates image by taking in random numbers/vectors. The image is then fed into the discrimnator along with other images taken from the actual dataset. It the nreturns probability between 0 and 1 of whether it is fake or not with 0 representing a fake and 1 representing an authentic image. Generator tries to get better at creating images and discriminator is trying to get better at identifying them as authentic or fake. The goal of the model is to get to the point where the generator can create images/input/features that cannot be identified as fake by the discrminator i.e. they look as real as belonging to the dataset that it is trained on.
# 

# <a id="p4"></a>
# ## Part 4 - More...

# Answer the following questions, with a target audience of a fellow Data Scientist:
# 
# - What do you consider your strongest area, as a Data Scientist?
# - What area of Data Science would you most like to learn more about, and why?
# - Where do you think Data Science will be in 5 years?
# - What are the treats posed by AI to our society?
# - How do you think we can counteract those threats? 
# - Do you think achieving General Artifical Intelligence is ever possible?
# 
# A few sentences per answer is fine - only elaborate if time allows.

# So far, I have had the most fun with neural networks and all the remarkable things you can do with them. I wouldn't say that is my strongest area but it is definitely the area I want to advance in and master. I would love to learn more about object detection and computer vision, partially because I want a Knight Rider like car that takes me places and partially because I can envision many practical applications of it from helping in surgeries to self-driving cars. I believe data science can really change the world and society for the better if put to good use. Right now, there is a lot of hype around the field and a lot of companies have to figure out how to use it in the best way possible. But if it is not used for malicious intent, it could produce great solution to a lot of problems that exist in our society like poverty, inequality, injustice, poor health, and many more. 
# 
# The question ofwhether AI poses threats to the society can only be answered if we know who is it controlled by. If it is facial recognition software to help police officers identify past convicts in a robbery, then it could be very beneficial. But if it to give them drones to spy on poor/crime ridden neighborhoods like they do in Baltimore, then it can be seen as a threat to personal space, freedom and privacy. Same goes for all the projects that big companies like Google, Facebook, Microsoft are working on. One of the biggest threats that you can envision is development of skynet like programs. We already have DARPA funding research on autonomous weapons and despite the outcry by activists demanding more regulation, it has been going on. 
# 
# The only way to counteract these and other unseen or unfathomable threats is to regulate the field by an overseeing authority. I would not recommend stopping the research or advances in the field of AI because as mentioned above, it could potentially benefit humanity in a lot of ways. But there need to be an overseeing committee that ensures that all the research that is being done is for the greater good. A committee that has authority to oversee projects going on anywhere in the world would be the best solution so there is no prisoner's dilemma for one country vs. another to use AI research for malicious intents. 
# 
# While I would love to have a Terminator like companion, I don't think it is entirely possible to the point where a machine can mimic every human emotion. If we want to think of ourselves as robots, which we are in many ways i.e. being told to conform to the society, going to school, getting a job, getting married, growing old. In a lot of ways, we are already like robots and sure, we can make any robots to do that. But having a machine that can perform all that plus mimic all the human emotions/subconscious is going to be complicated. While the machines can be great at one specific task like AlphaGo is great at Go and has surpassed humans in intelligence, having one specific machine to be great at everything has a long way to go. 

# ## Congratulations! 
# 
# Thank you for your hard work, and congratulations! You've learned a lot, and you should proudly call yourself a Data Scientist.
# 

# In[44]:


from IPython.display import HTML

HTML("""<iframe src="https://giphy.com/embed/26xivLqkv86uJzqWk" width="480" height="270" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/mumm-champagne-saber-26xivLqkv86uJzqWk">via GIPHY</a></p>""")


# In[ ]:




