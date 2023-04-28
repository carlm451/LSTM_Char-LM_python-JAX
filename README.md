# LSTM_Character_Language_Model
Character-level Long Short Term Memory Recurrent Neural Network implemented from scratch in Python


This repo contains various iterations of my work to implement a character level LSTM language model from the ground up all in python. 

The code here is an extenstion of the minimal level RNN python script 'min-char-rnn.py' written by Andrej Karpathy for his popular blog post explaining
the fundamental mechanics of RNNs, "The Unreasonable Effectiveness of Recurrent Neural Networks"

I wanted to understand the detailed workings of more elaborate versions of RNNS, where there are many more internal computations inside each unit and also where there
are stacked RNNS to make a deep network, so I set out to extend the 'min-char-rnn.py' and see how many features I could implement from the ground up. 

Extensions to the min-char-rnn.py model:

* RNN --> LSTM ( new forward pass/new BPTT pass)
* single layer --> custom number of layer num_layers
* adagrad --> ADAM optimizer from scratch
* sampling T=1 ---> sampling with temperature T if desired
* and more! 

as;dlkjadf ;

need to work on this more...


