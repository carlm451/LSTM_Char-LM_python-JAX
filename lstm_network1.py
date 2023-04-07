# class to implement custom character-level LSTM in Python

import numpy as np

class LSTM(object):
    
    def __init__(self,seq_length,hidden_size):

        self.seq_length=seq_length  # number of time steps to unroll the LSTM
        self.hidden_size=hidden_size # size of the hidden layer inside the LSTM

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

    def stochastic_gradient_descent(self,training_data,mini_batch_size=10,learning_rate=0.03):

        # training data is preprocessed string with entire text file for training

        chars = list(set(training_data))
        self.vocab_size = len(chars)

        #can now initialize the weights, since we know input data size

        self.weight_initializer()
        self.n_steps=0
        
        self.losses = []

        smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length

        smooth_loss_start = smooth_loss

        self.losses.append(smooth_loss)
    
        self.losses2 = []
        self.losses2.append(smooth_loss)

        self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
        self.ix_to_char = { i:ch for i,ch in enumerate(chars) }
        
        n=0

        batch_generator = self.get_mini_batch(training_data,mini_batch_size)

        current_loss = 0

        while n < 1e6:

            inputs,targets = next(batch_generator)

            current_loss = self.sgd_step(inputs,targets,mini_batch_size,learning_rate)

            smooth_loss = smooth_loss*0.999 + current_loss*0.001

            # sample from the model now and then
            if n % 5000 == 0:
                sample_ix = self.sample(self.hprev,self.cprev, np.random.randint(self.vocab_size), 20)
                txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
                print('----\n %s \n----' % (txt,))

            if n%5000==0:
                print(f'Loss: {smooth_loss:.4f}\tRelative: {100*smooth_loss/smooth_loss_start:.4f}')

            self.losses.append(smooth_loss)
            self.losses2.append(current_loss)

            n+=1

    def get_mini_batch(self,data,mini_batch_size):
        '''
        generator to continuous loop the data and pull out mini batches
        '''
        p = 0

        batch_character_size = mini_batch_size*(self.seq_length)

        while True:

            if p+batch_character_size+1>=len(data):
                p=0

                self.hprev = np.zeros((self.hidden_size,1))
                self.cprve = np.zeros((self.hidden_size,1))

            inputs,targets = [],[]

            for _ in range(mini_batch_size):

                inputs.append([self.char_to_ix[ch] for ch in data[p:p+self.seq_length]])
                targets.append([self.char_to_ix[ch] for ch in data[p+1:p+self.seq_length+1]])

                p += self.seq_length

            yield inputs,targets

    def sgd_step(self,inputs,targets,mini_batch_size,learning_rate):
        '''
        Forward pass, backward pass, and gradient update for single minibatch of fixed length sequences
        and target sequences

        inputs -- list of mini_batch_size lists , each of seq_length character indices
        targets -- same size as inputs, lists are shifted by +1
        '''

        # unpack params

        Wxc = self.params['Wxc']
        Wxu = self.params['Wxu']
        Wxf = self.params['Wxf']
        Wxo = self.params['Wxo']

        Whc = self.params['Whc']
        Whu = self.params['Whu']
        Whf = self.params['Whf']
        Who = self.params['Who']

        bc = self.params['bc']
        bu = self.params['bu']
        bf = self.params['bf']
        bo = self.params['bo']

        Why = self.params['Why']
        by = self.params['by']

        # caches for forward pass

        xs, hs, cs, c_tildes = {}, {}, {}, {}

        gamma_us, gamma_fs, gamma_os = {}, {}, {}

        ys, ps = {}, {}

        inputs = np.array(inputs)
        targets = np.array(targets)

        assert(inputs.shape==(mini_batch_size,self.seq_length))
        assert(targets.shape==(mini_batch_size,self.seq_length))

        hs[-1] = np.tile(self.hprev,(1,mini_batch_size))
        cs[-1] = np.tile(self.cprev,(1,mini_batch_size))

        loss = 0
        
        for t in range(self.seq_length):
            xs[t] = np.zeros((self.vocab_size,mini_batch_size))

            xs[t][inputs[:,t],np.arange(mini_batch_size)]=1  # batch of one-hot vectors for time t 

            zc = np.dot(Wxc,xs[t]) + np.dot(Whc,hs[t-1]) + bc  # linear activation for candidate cell state C~
            zu = np.dot(Wxu,xs[t]) + np.dot(Whu,hs[t-1]) + bu  # linear activation for update gate
            zf = np.dot(Wxf,xs[t]) + np.dot(Whf,hs[t-1]) + bf  # linear activation for forget gate
            zo = np.dot(Wxo,xs[t]) + np.dot(Who,hs[t-1]) + bo  # linear activation for output gate

            c_tildes[t] = np.tanh(zc) # canidate for new c state

            gamma_us[t] = sigmoid(zu)
            gamma_fs[t] = sigmoid(zf)
            gamma_os[t] = sigmoid(zo)

            cs[t] = np.tanh(np.multiply(c_tildes[t],gamma_us[t]) + np.multiply(cs[t-1],gamma_fs[t]))  # tanh here is import!!!

            hs[t] = np.multiply(cs[t],gamma_os[t]) # hidden state

            ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars

            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]),axis=0) # probabilities for next chars

            loss += -np.mean(np.log(ps[t][targets[:,t],np.arange(mini_batch_size)]))   #  ps[t] should be shape (vocab_size,mini_batch_size)
    
        #initialize gradients to zero

        dWxc,dWxu,dWxf,dWxo = np.zeros_like(Wxc), np.zeros_like(Wxu), np.zeros_like(Wxf), np.zeros_like(Wxo)
        dWhc,dWhu,dWhf,dWho = np.zeros_like(Whc), np.zeros_like(Whu), np.zeros_like(Whf), np.zeros_like(Who)
        dbc,dbu,dbf,dbo = np.zeros_like(bc), np.zeros_like(bu), np.zeros_like(bf), np.zeros_like(bo)
        dWhy,dby = np.zeros_like(Why), np.zeros_like(by)

        # tmp variables to accumulate gradients over the backprop -- see differentiation graph  
        dhnext, dcnext = np.zeros_like(hs[0]), np.zeros_like(cs[0])

        #backward pass
        
        for t in reversed(range(self.seq_length)):
            dy = np.copy(ps[t])

            dy[targets[:,t],np.arange(mini_batch_size)] -= 1 # backprop into y
    
            dWhy += np.dot(dy, hs[t].T)
            dby += np.sum(dy,axis=1,keepdims=True)

            dh = np.dot(Why.T, dy) + dhnext # backprop into h

            dc = np.multiply((1-cs[t]**2),np.multiply(gamma_os[t],dh) + dcnext) #backprop into c

            dcnext = np.multiply(gamma_fs[t],dc)

            dzc = np.multiply((1-c_tildes[t]**2),np.multiply(gamma_us[t],dc))  # backprop through tanh

            dzu = np.multiply(gamma_us[t]*(1-gamma_us[t]),np.multiply(c_tildes[t],dc))  # sigmoid prime

            dzf = np.multiply(gamma_fs[t]*(1-gamma_fs[t]),np.multiply(cs[t-1],dc))

            dzo = np.multiply(gamma_os[t]*(1-gamma_os[t]),np.multiply(cs[t],dh))

            dbc += np.sum(dzc,axis=1,keepdims=True)
            dbu += np.sum(dzu,axis=1,keepdims=True)
            dbf += np.sum(dzf,axis=1,keepdims=True)
            dbo += np.sum(dzo,axis=1,keepdims=True)

            dWhc += np.dot(dzc,hs[t-1].T)
            dWhu += np.dot(dzu,hs[t-1].T)
            dWhf += np.dot(dzf,hs[t-1].T)
            dWho += np.dot(dzo,hs[t-1].T)

            dWxc += np.dot(dzc,xs[t].T)
            dWxu += np.dot(dzu,xs[t].T)
            dWxf += np.dot(dzf,xs[t].T)
            dWxo += np.dot(dzo,xs[t].T)

            # four contributions to dhnext,one from each gate
            dhnext = np.dot(Whc.T,dzc) + np.dot(Whu.T,dzu) + np.dot(Whf.T,dzf) + np.dot(Who.T,dzo)
        
        grads = dict()

        grads['Wxc']=dWxc
        grads['Wxu']=dWxu
        grads['Wxf']=dWxf
        grads['Wxo']=dWxo

        grads['Whc']=dWhc
        grads['Whu']=dWhu
        grads['Whf']=dWhf
        grads['Who']=dWho

        grads['bc']=dbc
        grads['bu']=dbu
        grads['bf']=dbf
        grads['bo']=dbo

        grads['Why']=dWhy
        grads['by']=dby

        for parameter in grads.keys():
            np.clip(grads[parameter], -5, 5, out=grads[parameter]) # clip to mitigate exploding gradients

        self.grads=grads

        self.hprev = np.mean(hs[self.seq_length-1],axis=1,keepdims=True)
        self.cprev = np.mean(cs[self.seq_length-1],axis=1,keepdims=True)

        # perform parameter update with Adagrad
        for parameter in self.params.keys():
            
            dparam = self.grads[parameter]

            self.mems[parameter] += dparam * dparam

            mem = self.mems[parameter]
            
            self.params[parameter] += -learning_rate * dparam / (mini_batch_size * np.sqrt(mem + 1e-8)) # adagrad update

        return loss

    def sample(self,h, c, seed_ix, n):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """
        
        Wxc = self.params['Wxc']
        Wxu = self.params['Wxu']
        Wxf = self.params['Wxf']
        Wxo = self.params['Wxo']

        Whc = self.params['Whc']
        Whu = self.params['Whu']
        Whf = self.params['Whf']
        Who = self.params['Who']

        bc = self.params['bc']
        bu = self.params['bu']
        bf = self.params['bf']
        bo = self.params['bo']

        Why = self.params['Why']
        by = self.params['by']

        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):

            zc = np.dot(Wxc,x) + np.dot(Whc,h) + bc  # linear activation for candidate cell state C~
            zu = np.dot(Wxu,x) + np.dot(Whu,h) + bu  # linear activation for update gate
            zf = np.dot(Wxf,x) + np.dot(Whf,h) + bf  # linear activation for forget gate
            zo = np.dot(Wxo,x) + np.dot(Who,h) + bo  # linear activation for output gate

            c_tilde = np.tanh(zc)

            gamma_u = sigmoid(zu)
            gamma_f = sigmoid(zf)
            gamma_o = sigmoid(zo)

            c = np.tanh(np.multiply(c_tilde,gamma_u) + np.multiply(c,gamma_f))

            h = np.multiply(c,gamma_o) # hidden state

            y = np.dot(Why, h) + by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        
        return ixes

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

if __name__ == '__main__':

    seq_length = 3

    hidden_size = 20

    lstm = LSTM(seq_length,hidden_size)

    print(f'LSTM with {lstm.hidden_size} hidden units')
    print(f'Will train on character sequences of length {lstm.seq_length}')
    print('\n\n\n')

    training_data = 'abcdefghijklmnopqrstuvwxyz'*500

    print('Toy training data: ',training_data[:26*3])
    print('\n\n\n')

    lstm.stochastic_gradient_descent(training_data=training_data)
