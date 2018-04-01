![Figure 1-1](https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/200px-TensorFlowLogo.svg.png "Tensorflow")

# Exploring Recurrent Neural Networks in Tensorflow

**Basic RNN - simple_rnn.py**

Simple demonstration of RNN with only two time steps. For learning purposes. 

**Learning a waveform-like time series structure - time_series.py**

Demonstrates how a RNN can be used to learn a time-series like structure. 
Note that the RNN can produce predictions that map very closely to (a part of) the original waveform. 
The predictions were then used to synthesize a waveform based on the learning of the rnn model. 

**Creation of word embeddings, typically for NLP Applications - word2vec.py**

This isn't RNN, but is typical of an initial step for NLP RNN Applications. 
It is basically a 3-layer neural network, utilizing noise contrastive estimation for getting the loss function.   

Inputs -> Hidden Layer -> Outputs 

Most complete explanation I've found for this particular code : 
http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/

I also like 

https://towardsdatascience.com/learn-word2vec-by-implementing-it-in-tensorflow-45641adaf2ac
om/learn-word2vec-by-implementing-it-in-tensorflow-45641adaf2ac

for the hand-drawn diagrams that lent some clarity.
In word2vec, **T-SNE** was also used to visualize the word embeddings.


![Figure 1-2](https://blog.keras.io/img/keras-tensorflow-logo.jpg "Keras")

# Keras - Translation Using Encoder-Decoder Architecture

**Translation of English to French - keras_translation.py**

Keras is now fully compatible with Tensorflow and allows for high-level prototyping. 
Here we will explore how Keras can be used to build a translation system, and use it to
translate English to French. This is a simple char-by-char input model. A more advanced
would be a word-by-word input model, where the network would need to learn word embeddings
(ie. what you get from word2vec)


![Figure 1-3](https://cdn-images-1.medium.com/max/2000/1*nYptRUTtVd9xUjwL-cVL3Q.png " Encoder-Decoder Inference model architecture for NMT —image copyright @Ravindra Kompella")
Encoder-Decoder Inference model architecture for NMT —image copyright @Ravindra Kompella

Run tensorboard to visualize the graph. It's so much easier to use tensorboard in Keras.
````
tensorboard --logdir=logs/
````

Output
````
7680/8000 [===========================>..] - ETA: 0s - loss: 0.1292
7744/8000 [============================>.] - ETA: 0s - loss: 0.1293
7808/8000 [============================>.] - ETA: 0s - loss: 0.1293
7872/8000 [============================>.] - ETA: 0s - loss: 0.1294
7936/8000 [============================>.] - ETA: 0s - loss: 0.1296
8000/8000 [==============================] - 7s 869us/step - loss: 0.1296 - val_loss: 0.5832
-
Input sentence: ﻿Go.
Decoded sentence: Va !

-
Input sentence: Run!
Decoded sentence: Auplez les charss !

-
Input sentence: Run!
Decoded sentence: Auplez les charss !

-
Input sentence: Wow!
Decoded sentence: Ditas se l'arre !

````
Hmmm... after checking with Google Translate I found that that wasn't so accurate. Probably because
it was a character-by-character model, meaning that the french words could be spelt wrongly. 
Some characters could have been predicted wrongly.

**Generate Fake Donald Trump Tweets - fake_donald_trump.py**

A RNN model to generate fake donald trump tweets by basing it on his Twitter history. Data is retrieved from
http://www.trumptwitterarchive.com





