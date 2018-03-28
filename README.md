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

Most complete explanation I've found for this particular code : 
http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/

I also like 
https://towardsdatascience.com/learn-word2vec-by-implementing-it-in-tensorflow-45641adaf2ac

for the hand-drawn diagrams that lent some clarity.

