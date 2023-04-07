# class to implement custom character-level LSTM in Python

import numpy as np

class LSTM(object):
    
    def __init__(self,seq_length,hidden_size):

        self.seq_length=seq_length  # number of time steps to unroll the LSTM
        self.hidden_size=hidden_size # size of the hidden layer inside the LSTM

        self.loss = 0

        self.hprev = np.zeros((self.hidden_size,1))
        self.cprev = np.zeros((self.hidden_size,1))

    def weight_initializer(self):

        n_h = self.hidden_size
        n_x = self.vocab_size

        params = dict()

        params['Wxc'] = np.random.randn(n_h, n_x)*0.01 # input to cell state
        params['Wxu'] = np.random.randn(n_h, n_x)*0.01 # input to update
        params['Wxf'] = np.random.randn(n_h, n_x)*0.01 # input to forget
        params['Wxo'] = np.random.randn(n_h, n_x)*0.01 # input to output

        params['bc'] = np.zeros((n_h, 1)) # hidden bias
        params['bu'] = np.zeros((n_h, 1)) # forget bias
        params['bf'] = np.zeros((n_h, 1)) # update bias
        params['bo'] = np.zeros((n_h, 1)) # output bias

        params['Whc'] = np.random.randn(n_h,n_h)*0.01 # hidden to cell
        params['Whu'] = np.random.randn(n_h,n_h)*0.01 # hidden to update
        params['Whf'] = np.random.randn(n_h,n_h)*0.01 # hidden to forget
        params['Who'] = np.random.randn(n_h,n_h)*0.01 # hidden to output

        params['Why'] = np.random.randn(n_x, n_h)*0.01 # hidden to output
        params['by'] = np.zeros((n_x, 1)) # output bias

        self.params = params

        mems = dict()

        for parameter in self.params.keys():
            
            shape = params[parameter].shape

            mems[parameter] = np.zeros(shape)

        self.mems = mems

    def stochastic_gradient_descent(self,training_data,mini_batch_size,learning_rate):

        # training data is preprocessed string with entire text file for training

        chars = list(set(training_data))
        self.vocab_size = len(chars)

        #can now initialize the weights, since we know input data size

        self.weight_initializer()
        self.n_steps=0
    
        self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
        self.ix_to_char = { i:ch for i,ch in enumerate(chars) }
        
        n=0

        while n < 2:

            batch_generator = self.get_mini_batch(training_data,mini_batch_size)

            inputs,targets = next(batch_generator)

            print("Inputs: ",inputs)
            print("Targets: ",targets)

    def get_mini_batch(self,data,mini_batch_size):

        p = 0

        while p+self.seq_length+1<len(data):

            inputs,targets = [],[]

            for _ in range(mini_batch_size):

                inputs.append([self.char_to_ix[ch] for ch in data[p:p+self.seq_length]])
                targets.append([self.char_to_ix[ch] for ch in data[p+1:p+self.seq_length+1]])

                p += self.seq_length

            yield inputs,targets
        

if __name__ == '__main__':

    seq_length = 3

    hidden_size = 10

    lstm = LSTM(seq_length,hidden_size)

    print(type(lstm),lstm.seq_length,lstm.hidden_size)

    training_data = 'abcdefghijklmnopqrstuv'

    lstm.stochastic_gradient_descent(training_data=training_data,mini_batch_size=2,learning_rate=0.001)

    print(lstm.params['Wxc'].shape,lstm.mems['Wxc'].shape)

    print(lstm.params.keys())
