"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License

Modified by Carl Merrigan, to test on .txt files from Project Gutenberg
"""
import numpy as np
import os
from pprint import pprint

# choose a book from the available project gutenberg text files

directory = "books/"
dir_list = os.listdir(directory)

print("Here are the books available to read: ")
pprint(dir_list)
print()

book = input("Book to read ? ")
filename = directory + book + '.txt'
print(f"Reading {filename}")
print()

if not os.path.isfile(filename):
    print(f"No text by the title: {book}")
    exit()

print()

data = open(filename,'r').read()
print(f'Original # of characters = {len(data)}')

# Search for occurences of the title on a line by itself, case insensitive

import re

match = re.search(r'\*\*\* START .+ \*\*\*',data)

if match != None:

    s1 = match.start()
    e1 = match.end()

    print(data[s1:e1])
    print()

match = re.search(r'\*\*\* END .+ \*\*\*',data)

if match != None:

    s2 = match.start()
    e2 = match.end()

    print(data[s2:e2])

print("Clipping content from Gutenberg text file... ")

data = data[e1+30:s2-30]

data = data.replace('\n',' ')

print("Text data ready to process...")

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('The book has %d characters, %d unique.' % (data_size, vocab_size))

proceed = input("Proceed with Character RNN Training? (enter: quit, 'ok': continue):  ")

if not proceed:
    exit()

#data I/O
#chars = list(set(data))
#data_size, vocab_size = len(data), len(chars)
#print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-2

print(f"Using {hidden_size} hidden units, sequence lengths: {seq_length}, and learning rate: {learning_rate}")

# model parameters

Wxc = np.random.randn(hidden_size, vocab_size)*0.01 # input to cell state
Wxu = np.random.randn(hidden_size, vocab_size)*0.01 # input to update
Wxf = np.random.randn(hidden_size, vocab_size)*0.01 # input to forget
Wxo = np.random.randn(hidden_size, vocab_size)*0.01 # input to output

bc = np.zeros((hidden_size, 1)) # hidden bias
bu = np.zeros((hidden_size, 1)) # forget bias
bf = np.zeros((hidden_size, 1)) # update bias
bo = np.zeros((hidden_size, 1)) # output bias

Whc = np.random.randn(hidden_size,hidden_size) # hidden to cell
Whu = np.random.randn(hidden_size,hidden_size) # hidden to update
Whf = np.random.randn(hidden_size,hidden_size) # hidden to forget
Who = np.random.randn(hidden_size,hidden_size) # hidden to output

Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
by = np.zeros((vocab_size, 1)) # output bias

params = [Wxc,Wxu,Wxf,Wxo,Whc,Whu,Whf,Who,bc,bu,bf,bo,Why,by]

def lossFun(inputs, targets, hprev, cprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, cs, c_tildes = {}, {}, {}, {}

  gamma_us, gamma_fs, gamma_os = {}, {}, {}

  ys, ps = {}, {}

  hs[-1] = np.copy(hprev)
  cs[-1] = np.copy(cprev)
  
  loss = 0
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1

    zc = np.dot(Wxc,xs[t]) + np.dot(Whc,hs[t-1]) + bc  # linear activation for candidate cell state C~
    zu = np.dot(Wxu,xs[t]) + np.dot(Whu,hs[t-1]) + bu  # linear activation for update gate
    zf = np.dot(Wxf,xs[t]) + np.dot(Whf,hs[t-1]) + bf  # linear activation for forget gate
    zo = np.dot(Wxo,xs[t]) + np.dot(Who,hs[t-1]) + bo  # linear activation for output gate

    c_tildes[t] = np.tanh(zc) # canidate for new c state

    gamma_us[t] = np.sigmoid(zu)
    gamma_fs[t] = np.sigmoid(zf)
    gamma_os[t] = np.sigmoid(zo)

    cs[t] = np.multiply(c_tildes[t],gamma_us[t]) + np.multiply(cs[t-1],gamma_fs[t])

    hs[t] = np.multiply(cs[t],gamma_os[t]) # hidden state
    
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars

    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  
  dWxc,dWxu,dWxf,dWxo = np.zeros_like(Wxc), np.zeros_like(Wxu), np.zeros_like(Wxf), np.zeros_like(Wxo)
  dWhc,dWhu,dWhf,dWho = np.zeros_like(Whc), np.zeros_like(Whu), np.zeros_like(Whf), np.zeros_like(Who)
  dbc,dbu,dbf,dbo = np.zeros_like(bc), np.zeros_like(bu), np.zeros_like(bf), np.zeros_like(bo)
  dWhy,dby = np.zeros_like(Why), np.zeros_like(by)
  
  dhnext, dcnext = np.zeros_like(hs[0]), np.zeros_like(cs[0])

  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    
    dWhy += np.dot(dy, hs[t].T)
    dby += dy

    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    
    dc = np.dot(gamma_os[t],dh) + dcnext #backprop into c 

    dcnext = np.dot(gamma_fs[t],dc)

    dzc = np.multiply((1-c_tildes[t]**2),np.multiply(gamma_us[t],dc))  # backprop through tanh

    dzu = np.multiply(gamma_us[t]*(1-gamma_us[t]),np.multiply(c_tildes[t],dc))  # sigmoid prime

    dzf = np.multiply(gamma_fs[t]*(1-gamma_fs[t]),np.multiply(cs[t-1],dc))

    dzo = np.multiply(gamma_os[t]*(1-gamma_os[t]),np.multiply(cs[t],dh))

    dbc += dzc
    dbu += dzu
    dbf += dzf
    dbo += dzo

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

  grads = [dWxc,dWxu,dWxf,dWxo,dWhc,dWhu,dWhf,dWho,dbc,dbu,dbf,dbo,dWhy,dby]

  for dparam in grads:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, grads, hs[len(inputs)-1], cs[len(inputs)-1]

def sample(h, c, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
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

    c = np.multiply(c_tilde,gamma_u) + np.multiply(c,gamma_f)

    h = np.multiply(c,gamma_o) # hidden state

    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

n, p = 0, 0

# memory variables for Adagrad

mWxc,mWxu,mWxf,mWxo = np.zeros_like(Wxc), np.zeros_like(Wxu), np.zeros_like(Wxf), np.zeros_like(Wxo)
mWhc,mWhu,mWhf,mWho = np.zeros_like(Whc), np.zeros_like(Whu), np.zeros_like(Whf), np.zeros_like(Who)
mbc,mbu,mbf,mbo = np.zeros_like(bc), np.zeros_like(bu), np.zeros_like(bf), np.zeros_like(bo)
mWhy,mby = np.zeros_like(Why), np.zeros_like(by)

mems=[mWxc,mWxu,mWxf,mWxo,mWhc,mWhu,mWhf,mWho,mbc,mbu,mbf,mbo,mWhy,mby]

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

while n<1e6:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    cprev = np.zeros((hidden_size,1)) 

    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 5000 == 0:
    sample_ix = sample(hprev,cprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt,))

  # forward seq_length characters through the net and fetch gradient
  loss, grads, hprev, cprev = lossFun(inputs, targets, hprev, cprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 1000 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip(params, 
                                grads, 
                                mems):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter
