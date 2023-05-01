"""set of functions to build custom deep lstm character language models
##
###
####
description here... 

"""

import jax.numpy as jnp
from jax import grad, jit
from jax import random

from textwrap import wrap

from copy import deepcopy


def initialize_lstm_weights(key,n_h,n_x):
    """
    Initializes lstm layer weights with normal distribution

    Args:
        key (PRNG Key): Key controlling jax.random random number generation
        n_h (int): Dimension of the LSTM hidden layer
        n_x (int): Dimension of the input embedding vectors

    Returns:
        params (dict): Collection of all LSTM layer parameters

        Wxc,Wxu,Wxf,Wxo (jax array): matrices shape=(n_h,n_x) act on batchs
                                     of input column vectors (batch_size,n_x)

        Whc,Whu,Whf,Who (jax array): matrices shape=(n_h,n_h) act on hidden states

        bc,bu,bf,bo (jax array): bias vectors shape=(n_h,1)

        grad_mems(dict): Collection of memory arrays for gradients , needed for ADAM algorithm
        sqrd_mems(dict): Collection of memory arrays for squared gradients
    """

    subkeys=random.split(key,9) # need to call random.noraml with new key each time

    params, grad_mems, sqrd_mems = dict(),dict(),dict()

    params['Wxc'] = random.normal(subkeys[1],(n_h,n_x))*0.01 # input to cell state
    params['Wxu'] = random.normal(subkeys[2],(n_h,n_x))*0.01 # input to update
    params['Wxf'] = random.normal(subkeys[3],(n_h,n_x))*0.01 # input to forget
    params['Wxo'] = random.normal(subkeys[4],(n_h,n_x))*0.01 # input to output

    params['bc'] = jnp.zeros((n_h, 1)) # hidden bias
    params['bu'] = jnp.zeros((n_h, 1)) # forget bias
    params['bf'] = jnp.zeros((n_h, 1)) # update bias
    params['bo'] = jnp.zeros((n_h, 1)) # output bias

    params['Whc'] = random.normal(subkeys[5],(n_h,n_h))*0.01 # hidden to cell
    params['Whu'] = random.normal(subkeys[6],(n_h,n_h))*0.01 # hidden to update
    params['Whf'] = random.normal(subkeys[7],(n_h,n_h))*0.01 # hidden to forget
    params['Who'] = random.normal(subkeys[8],(n_h,n_h))*0.01 # hidden to output

    for parameter in params.keys():

        shape = params[parameter].shape

        grad_mems[parameter]=jnp.zeros(shape)
        sqrd_mems[parameter]=jnp.zeros(shape)

    return params, grad_mems, sqrd_mems

def initialize_dense_weights(key,n_h,n_x):
    """
    Initializes final dense layer weights for character probabilities 

    Args:
        key (PRNG Key): Key controlling jax.random random number generation
        n_h (int): Dimension of the LSTM hidden layer
        n_x (int): Vocab size for softmatx outputs 

    Returns:
        params (dict): Collection of all LSTM layer parameters

        Why (jax array): matrix shape=(n_x,n_h) maps hidden states to output
                         probabilities over characters in the vocab

        by (jax array): bia vectors shape=(n_x,1)

        grad_mems(dict): Collection of memory arrays for gradients , needed for ADAM algorithm
        sqrd_mems(dict): Collection of memory arrays for squared gradients
    """

    key,subkey=random.split(key)

    params,grad_mems,sqrd_mems = dict(),dict(),dict()

    params['Why'] = random.normal(subkey,(n_x,n_h))*0.01 # hidden to output
    params['by'] = jnp.zeros((n_x, 1)) # output bias

    for parameter in params.keys():

        shape = params[parameter].shape

        grad_mems[parameter]=jnp.zeros(shape)
        sqrd_mems[parameter]=jnp.zeros(shape)

    return params, grad_mems, sqrd_mems

def get_mini_batch(mini_batch_size,seq_length,char_to_ix,data):
    '''
    Generator that continuously yields mini-batches of text sequences

    Args:
        mini_batch_size (int): Number of sequences each mini batch
        seq_length (int): Number of characters per sequence
        char_to_ix (dict): Dict mapping characters to their indices

        data (str): String containing all the characters in the data set
        
    Returns:
        inputs (jax array): batch of sequence indices shape = (mini_batch_size,seq_length)
        targets (jax array): shifted sequences with next character targets
        
        epoch (int): counter tracking number of times have looped over entire data set 
    '''

    p = 0
    epoch = 0

    batch_character_size = mini_batch_size*(seq_length)

    while True:

        if p+batch_character_size+1>=len(data):
            p=0
            epoch = epoch + 1

            # need to reset hprev,cprev if it loops?

        inputs,targets = [],[]

        for _ in range(mini_batch_size):

            inputs.append([char_to_ix[ch] for ch in data[p:p+seq_length]])
            targets.append([char_to_ix[ch] for ch in data[p+1:p+seq_length+1]])

            p += seq_length

        inputs = jnp.array(inputs)
        targets = jnp.array(targets)

        yield inputs,targets,epoch

def encode_inputs(inputs,vocab_size):
    '''
    One-hot encodes batches of characater sequences

    Args:
        inputs (jax array): batch of characters shape=(mini_batch_size,seq_length)
        vocab_size (int): number of characters in the vocab
    Returns:
        xs (dict): dictionary of jax arrays, indexed by time t in [0,1,...,seq_length]
        xs[t]: jax array at time t, shape = (vocab_size,mini_batch_size)
    '''

    xs = {}

    xs[-1] = None  # need hs and xs dicts to have same length = seq_length + 1

    mini_batch_size = inputs.shape[0]
    seq_length = inputs.shape[1]
    
    for t in range(seq_length):
        xs[t] = jnp.zeros((vocab_size,mini_batch_size))
        xs[t] = xs[t].at[inputs[:,t],jnp.arange(mini_batch_size)].set(1) # batch of one-hot vectors for time t

    return xs

def sigmoid(z):
    # sigmoid activation for LSTM gates
    return 1.0/(1.0 + jnp.exp(-z))

def softmax(y):
    #computes softmax probabilities over characters
    return jnp.exp(y) / jnp.sum(jnp.exp(y),axis=0)

def lstm_forward(inputs,hprev,cprev,params):
    '''
    Computes forward pass for single LSTM layer

    Args:
        inputs (dict): dict of jax arrays each of shape (input_size,mini_batch_size)
        hprev (jax array): column vector (n_h,1) for starting hidden state
        cprev (jax array): column vector (n_h,1) for starting cell state
        params (dict): dictionary of matrices W and biases b for this layer 
    Returns:
        states (tuple): Cache of variables needed layer for backprop 
                        xs (dict): input vectors each time step
                        hs (dict): hidden states
                        cs (dict): cell states
                        c_tildes (dict): candidate cell states 
        gates (tuple): Cache of the LSTM gates 
                        gamma_us (dict): update gates
                        gamma_fs (dict): forget gates
                        gamma_os (dict): output gates
    '''

    # dims
    mini_batch_size = inputs[0].shape[1]
    seq_length = len(inputs) - 1
    #print("SEQ LENGTH ",seq_length)

    vocab_size = params['Wxc'].shape[1]

    # unpack params
    Wxc = params['Wxc']
    Wxu = params['Wxu']
    Wxf = params['Wxf']
    Wxo = params['Wxo']

    Whc = params['Whc']
    Whu = params['Whu']
    Whf = params['Whf']
    Who = params['Who']

    bc = params['bc']
    bu = params['bu']
    bf = params['bf']
    bo = params['bo']

    xs = inputs  # inputs should be given as dict with seq_length entries, each item = (n_x,m) array of vectors

    hs, cs, c_tildes = {},{},{}

    gamma_us, gamma_fs, gamma_os = {}, {}, {}

    hs[-1] = jnp.tile(hprev,(1,mini_batch_size))
    cs[-1] = jnp.tile(cprev,(1,mini_batch_size))

    for t in range(seq_length):

        zc = jnp.dot(Wxc,xs[t]) + jnp.dot(Whc,hs[t-1]) + bc  # linear activation for candidate cell state C~
        zu = jnp.dot(Wxu,xs[t]) + jnp.dot(Whu,hs[t-1]) + bu  # linear activation for update gate
        zf = jnp.dot(Wxf,xs[t]) + jnp.dot(Whf,hs[t-1]) + bf  # linear activation for forget gate
        zo = jnp.dot(Wxo,xs[t]) + jnp.dot(Who,hs[t-1]) + bo  # linear activation for output gate

        c_tildes[t] = jnp.tanh(zc) # canidate for new c state

        gamma_us[t] = sigmoid(zu)
        gamma_fs[t] = sigmoid(zf)
        gamma_os[t] = sigmoid(zo)

        cs[t] = jnp.tanh(jnp.multiply(c_tildes[t],gamma_us[t]) + jnp.multiply(cs[t-1],gamma_fs[t]))  # tanh here is import!!!

        hs[t] = jnp.multiply(cs[t],gamma_os[t]) # hidden state

    gates = (gamma_us,gamma_fs,gamma_os)

    states = (xs,hs,cs,c_tildes)

    return states, gates

def loss_function(targets,hidden_states,params):
    '''
    Computes categorical cross entropy loss over predicted characters 

    Args:
        targets (jax array): batch of target sequences shape=(mini_batch_size,seq_length)
        hidden_states (dict): collection of hidden states from last LSTM layer
        params (dict): jax arrays with layer weights
    Returns:
        loss (float): categorical cross entropy loss averaged over batch sequences
        p_cache (tuple): cache with logits and softmax probabilies 
                        ys (dict): set of unnormalized log probs on characters
                        ps (dict): set of softmax probabilities 
    '''

    mini_batch_size=targets.shape[0]
    seq_length = targets.shape[1]

    Why = params['Why']
    by = params['by']

    ys, ps = {},{}

    loss = 0

    for t in range(seq_length):

        ys[t] = jnp.dot(Why, hidden_states[t]) + by # unnormalized log probabilities for next chars

        ps[t] = softmax(ys[t]) # probabilities for next chars  #  ps[t] should be shape (vocab_size,mini_batch_size)

        loss += jnp.mean(jnp.log(jnp.sum(jnp.exp(ys[t]),axis=0)) - ys[t][targets[:,t],jnp.arange(mini_batch_size)])
        #loss += -jnp.mean(jnp.log(ps[t][targets[:,t],jnp.arange(mini_batch_size)]))

    p_cache = (ys,ps)

    return loss, p_cache

def loss_backward(targets,probs_cache,final_hidden_states,dense_params):
    '''
    Backpropagation through final loss layer

    Args:
        targets (jax_array): batch of target indices shape=(mini_batch_size,seq_length)
        probs_cache (tuple): (ys,hs) logits and softmax probabilities 
        final_hidden_states (dict): hidden states output by last LSTM layer
        dense_params (dict): parameters for the loss layer
    Returns:
        dhs (dict): cache of hidden state grads to pass backwards
        grads (dict): dictionary with parameter gradients for this layer
    '''

    #unpack caches
    ys,ps = probs_cache
    hs = final_hidden_states  # dictionary with hidden states from last layer

    #weights for dense output to softmax layer
    Why = dense_params['Why']
    by = dense_params['by']

    # dims
    mini_batch_size = targets.shape[0]  # targets = (m,seq_length) rows of tokens each sample
    seq_length = targets.shape[1]

    vocab_size = Why.shape[0]

    #initialize grads
    dWhy,dby = jnp.zeros_like(Why), jnp.zeros_like(by)

    #cache to pass gradients back to lstm_backwards
    dhs = {}

    #backward pass
    for t in reversed(range(seq_length)):
        dy = jnp.copy(ps[t])

        dy = dy.at[targets[:,t],jnp.arange(mini_batch_size)].add(-1) #backprop into y

        dWhy += jnp.dot(dy, hs[t].T)
        dby += jnp.sum(dy,axis=1,keepdims=True)

        dhs[t] = jnp.dot(Why.T, dy)

    grads = dict()

    grads['Why']=dWhy
    grads['by']=dby

    for parameter in grads.keys():
        grads[parameter] = jnp.clip(grads[parameter], -5, 5) # clip to mitigate exploding gradients

    return dhs, grads

def lstm_layer_backward(dh_next_layer,gates_cache,states_cache,params):
    '''
    Computes backprop for a single LSTM layer 
    
    Args:
        dh_next_layer (dict): cache of intermediate derivatives from higher layer
        gates_cache (dict): gates cache from forward pass of this layer
        states_cache (dict): cache of foward pass variables this layer
        params (dict): set of weights W biases b for this layer
    Returns:
        dh_previous_layer (dict): cache of hidden state derivatives to pass backwards
        grads (dict): dictionary of gradients corresponding to params
    '''
    #unpack caches

    gamma_us,gamma_fs,gamma_os = gates_cache

    xs,hs,cs,c_tildes = states_cache

    # dh_next_layer # dictionary of derivatives from backpass of higher layer 

    # dims
    mini_batch_size = hs[-1].shape[1]  # note -1 is dict key for initilized h state , not last 
    seq_length = len(xs)-1 # make sure this works every layer ?

    vocab_size = params['Wxc'].shape[1]

    # unpack parameters

    Wxc = params['Wxc']
    Wxu = params['Wxu']
    Wxf = params['Wxf']
    Wxo = params['Wxo']

    Whc = params['Whc']
    Whu = params['Whu']
    Whf = params['Whf']
    Who = params['Who']

    bc = params['bc']
    bu = params['bu']
    bf = params['bf']
    bo = params['bo']

    #initialize gradients to zero

    dWxc,dWxu,dWxf,dWxo = jnp.zeros_like(Wxc), jnp.zeros_like(Wxu), jnp.zeros_like(Wxf), jnp.zeros_like(Wxo)
    dWhc,dWhu,dWhf,dWho = jnp.zeros_like(Whc), jnp.zeros_like(Whu), jnp.zeros_like(Whf), jnp.zeros_like(Who)
    dbc,dbu,dbf,dbo = jnp.zeros_like(bc), jnp.zeros_like(bu), jnp.zeros_like(bf), jnp.zeros_like(bo)
    
    # tmp variables to accumulate gradients over the backprop -- see differentiation graph
    dhnext, dcnext = jnp.zeros_like(hs[0]), jnp.zeros_like(cs[0])

    #need dictionary to pass dh derivative each t to earlier layer

    dh_previous_layer = {}

    #backward pass

    for t in reversed(range(seq_length)):
        
        dh = dh_next_layer[t] + dhnext # backprop into h

        dc = jnp.multiply((1-cs[t]**2),jnp.multiply(gamma_os[t],dh) + dcnext) #backprop into c

        dcnext = jnp.multiply(gamma_fs[t],dc)

        dzc = jnp.multiply((1-c_tildes[t]**2),jnp.multiply(gamma_us[t],dc))  # backprop through tanh

        dzu = jnp.multiply(gamma_us[t]*(1-gamma_us[t]),jnp.multiply(c_tildes[t],dc))  # sigmoid prime

        dzf = jnp.multiply(gamma_fs[t]*(1-gamma_fs[t]),jnp.multiply(cs[t-1],dc))

        dzo = jnp.multiply(gamma_os[t]*(1-gamma_os[t]),jnp.multiply(cs[t],dh))

        dbc += jnp.sum(dzc,axis=1,keepdims=True)
        dbu += jnp.sum(dzu,axis=1,keepdims=True)
        dbf += jnp.sum(dzf,axis=1,keepdims=True)
        dbo += jnp.sum(dzo,axis=1,keepdims=True)

        dWhc += jnp.dot(dzc,hs[t-1].T)
        dWhu += jnp.dot(dzu,hs[t-1].T)
        dWhf += jnp.dot(dzf,hs[t-1].T)
        dWho += jnp.dot(dzo,hs[t-1].T)

        dWxc += jnp.dot(dzc,xs[t].T)
        dWxu += jnp.dot(dzu,xs[t].T)
        dWxf += jnp.dot(dzf,xs[t].T)
        dWxo += jnp.dot(dzo,xs[t].T)

        # four contributions to dhnext,one from each gate
        dhnext = jnp.dot(Whc.T,dzc) + jnp.dot(Whu.T,dzu) + jnp.dot(Whf.T,dzf) + jnp.dot(Who.T,dzo)

        dh_previous_layer[t] = jnp.dot(Wxc.T,dzc) + jnp.dot(Wxu.T,dzu) + jnp.dot(Wxf.T,dzf) + jnp.dot(Wxo.T,dzo)

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

    for parameter in grads.keys():
        grads[parameter] = jnp.clip(grads[parameter], -5, 5) # clip to mitigate exploding gradients

    return dh_previous_layer, grads

@jit  # decorator to jit compile the function for faster execution 
def sgd_step_adam(current_step,inputs,targets,h_inputs,c_inputs,all_params,all_grads_mems,all_sqrd_mems,beta1,beta2,learning_rate):
    '''
    Performs a single optimization step using the ADAM algorithm 

    Args:
        current_step (int): counter of the steps performed so far during training
        inputs (jax array): batch of input sequences shape=(mini_batch_size,seq_length)
        targets (jax array): batch of target sequences shape=(mini_batch_size,seq_length)
        h_inputs (dict): dictionary indexed by layer of initial hidden inputs, layers number 0,1,...,num_layers-1
        c_inputs (dict): dict of intial cell state inputs each layer

        all_params (dict): collection of weights, num_layers lstm layers + final dense loss layer

                            all_params[0] - dictionary with 12 matrices for lstm layer 0
                            all_params[1] - second lstm layer
                            ...
                            all_params[num_layers-1] - last lstm layer
                            all_params[num_layers] - Why, by params of the output layer
                    
        all_grads (dict): running average of gradients each layer
        all_sqrd_memes (dict): running average of squared gradients each layer 

        beta1 (float): ADAM parameter for gradient averaging
        beta2 (float): ADAM parameter for squared gradient averaging
        learning_rate (float): learing rate for gradient descent step
    Returns:
        loss (float): categorical cross entropy loss for this step
        params_cache (tuple): shape = (params, grads, sqrd_grads) --> holds all the updated parameters
                             after single ADAM step
        hidden_cache (tuple): shape = (h_inputs,c_inputs) ---> final hidden states are saved to pass as inputs for next 
                                optimization step 
    '''
    print('Adam Step Tracing...') # won't appear in jit compiled function after first call 

    # extract dimensions
    mini_batch_size = inputs.shape[0]
    seq_length = inputs.shape[1]

    num_layers = len(all_params)-1  # this is number of lstm layers 

    vocab_size = all_params[0]['Wxc'].shape[1]

    n = current_step

    #one-hot encode the input batch

    layer_inputs = {}

    layer_inputs[0] = encode_inputs(inputs,vocab_size)

    s_cache_dict,g_cache_dict = {},{}

    dh_cache_dict = {}
    all_grads_dict = {} # holds gradients for parameters each layer 0,1,..num_layers-1 = LSTM and num_layers = dense 

    h_inputs_next = {} # need to pass final hidden states to next sgd_step
    c_inputs_next = {}

    #forwad pass
    for l in range(num_layers):

        h = h_inputs[l]
        c = c_inputs[l]

        x = layer_inputs[l]

        layer_params = all_params[l]

        s_cache_dict[l], g_cache_dict[l] = lstm_forward(x,h,c,layer_params)

        layer_inputs[l+1] = s_cache_dict[l][1]  # hidden states from this layer

        #h_inputs_next[l]=jnp.mean(layer_inputs[l+1][seq_length-1],axis=1,keepdims=True)
        #c_inputs_next[l]=jnp.mean(layer_inputs[l+1][seq_length-1],axis=1,keepdims=True)

        h_inputs_next[l]=jnp.expand_dims(layer_inputs[l+1][seq_length-1][:,-1],axis=1)
        c_inputs_next[l]=jnp.expand_dims(layer_inputs[l+1][seq_length-1][:,-1],axis=1)

    #compute loss
    loss,p_cache = loss_function(targets,layer_inputs[num_layers],all_params[num_layers])
  
    #loss backward 
    dh_cache_dict[num_layers], all_grads_dict[num_layers] = loss_backward(targets,p_cache,s_cache_dict[num_layers-1][1],all_params[num_layers])

    #lstm backwards
    for l in reversed(range(num_layers)):

        dh_cache_dict[l],all_grads_dict[l] = lstm_layer_backward(dh_cache_dict[l+1],g_cache_dict[l],s_cache_dict[l],all_params[l])

    new_all_params = deepcopy(all_params)
    new_all_grads_mems = deepcopy(all_grads_mems)
    new_all_sqrd_mems = deepcopy(all_sqrd_mems)

    #ADAM for all layers

    #loop throuh lstm layers = 0,1,...num_layers-1

    # dense layer = num_layers

    for l in range(num_layers+1):

        # perform parameter update with ADAM 
        for parameter in new_all_params[l].keys():

            dparam = all_grads_dict[l][parameter] / mini_batch_size

            new_all_grads_mems[l][parameter] = beta1*new_all_grads_mems[l][parameter] + (1-beta1)*dparam

            new_all_sqrd_mems[l][parameter] = beta2*new_all_sqrd_mems[l][parameter] + (1-beta2)*dparam*dparam

            grad_hat = new_all_grads_mems[l][parameter] / (1-beta1**(n+1))
            sqrd_hat = new_all_sqrd_mems[l][parameter] / (1-beta2**(n+1))

            new_all_params[l][parameter] += -learning_rate * grad_hat / (jnp.sqrt(sqrd_hat + 1e-8)) # ADAM update

    params_cache = (new_all_params,new_all_grads_mems,new_all_sqrd_mems)

    hidden_cache = (h_inputs_next,c_inputs_next)
    
    return loss, params_cache, hidden_cache

@jit 
def validation_loss(inputs,targets,h_inputs,c_inputs,all_params):
    '''
    Forward pass to get loss on validation data 

    Args: 
        inputs (jax array): batch of sequences (mini_batch,seq_length)
        targets (jax array): batch of sequences (mini_batch,seq_length)
        h_inputs (dict): hidden state inputs each lstm layer
        c_inputs (dict): cell state inputs each layer
    Returns: 
        val_loss (float): cross entropy loss on batch of validation data
    '''
    print('Val loss tracing...') # won't appear in jit compiled function 

    num_layers = len(all_params)-1

    vocab_size = all_params[0]['Wxc'].shape[1]

    layer_inputs={}

    layer_inputs[0] = encode_inputs(inputs,vocab_size)

    for l in range(num_layers):

        h = h_inputs[l]
        c = c_inputs[l]

        x = layer_inputs[l]

        layer_params = all_params[l]

        s_cache, _ = lstm_forward(x,h,c,layer_params)

        layer_inputs[l+1]=s_cache[1] #hidden outputs this layer

    val_loss,val_p_cache = loss_function(targets,layer_inputs[num_layers],all_params[num_layers])

    return val_loss

def sample(seed_ix,n,key,h_inputs,c_inputs,all_params,temperature=1.0):
    '''
    Sample from the model starting from single character 

    Args:
        seed_ix (int): index for starting character
        n (int): number of characters to sample
        key (PRNG key): key for jax random generation
        
        h_inputs (dict): dict of hidden state inputs at each layer
        c_inputs (dict): cell state inputs each layer

        all_params (dict): collection of layer weights indexed by layer
        
        temperature (dict): tune the softmax probabilities when sampling 
    Returns:
        ixes (list): list of sampled character indices 
    '''

    num_layers = len(all_params)-1

    vocab_size = all_params[0]['Wxc'].shape[1]

    #unpack params for output layer
    Why = all_params[num_layers]['Why']
    by = all_params[num_layers]['by']

    x = jnp.zeros((vocab_size, 1))
    x = x.at[seed_ix].set(1)

    layer_inputs = {}

    layer_inputs[0] = x

    xs_layers = {}
    hs_layers = {}
    cs_layers = {}

    for t in range(n+1):
        xs_layers[t]={}
        hs_layers[t]={}
        cs_layers[t]={}

    hs_layers[0]=h_inputs
    cs_layers[0]=c_inputs

    xs_layers[0][0] = x

    ixes = [seed_ix]
    for t in range(n):

        for l in range(num_layers):
            
            h = hs_layers[t][l]
            c = cs_layers[t][l]

            x = xs_layers[t][l]

            Wxc = all_params[l]['Wxc']
            Wxu = all_params[l]['Wxu']
            Wxf = all_params[l]['Wxf']
            Wxo = all_params[l]['Wxo']

            Whc = all_params[l]['Whc']
            Whu = all_params[l]['Whu']
            Whf = all_params[l]['Whf']
            Who = all_params[l]['Who']

            bc = all_params[l]['bc']
            bu = all_params[l]['bu']
            bf = all_params[l]['bf']
            bo = all_params[l]['bo']    

            zc = jnp.dot(Wxc,x) + jnp.dot(Whc,h) + bc  # linear activation for candidate cell state C~
            zu = jnp.dot(Wxu,x) + jnp.dot(Whu,h) + bu  # linear activation for update gate
            zf = jnp.dot(Wxf,x) + jnp.dot(Whf,h) + bf  # linear activation for forget gate
            zo = jnp.dot(Wxo,x) + jnp.dot(Who,h) + bo  # linear activation for output gate

            c_tilde = jnp.tanh(zc)

            gamma_u = sigmoid(zu)
            gamma_f = sigmoid(zf)
            gamma_o = sigmoid(zo)

            cs_layers[t+1][l] = jnp.tanh(jnp.multiply(c_tilde,gamma_u) + jnp.multiply(c,gamma_f))

            hs_layers[t+1][l] = jnp.multiply(cs_layers[t+1][l],gamma_o) # hidden state

            xs_layers[t][l+1] = hs_layers[t+1][l]

        y = jnp.dot(Why,xs_layers[t][l+1]) + by

        p = softmax(y/temperature)

        key,subkey = random.split(key)  #use key to split, subkey for next random number

        ix = random.choice(subkey,vocab_size,p=p.reshape(-1,))

        x_new = jnp.zeros((vocab_size, 1))
        x_new = x_new.at[ix].set(1)

        #ixes.append(int(ix))
        ixes.append(int(ix))

        xs_layers[t+1][0] = x_new

    return ixes

def train_character_lstm(seq_length,
                         hidden_sizes,
                         mini_batch_size,
                         learning_rate,
                         total_steps,
                         steps_sample_freq,
                         key,
                         training_data,
                         validation_data=None,
                         beta1=0.9,
                         beta2=0.999):
    '''
    Constructs and trains the lstm network on a body of text 

    Args:
        seq_length (int): number of character per training sequence
        hidden_sizes (list): list of layer sizes for num_layers=len(hidden_sizes) LSTM layers
        mini_batch_size (int): size of batches for each training step
        learning_rate (float): learning rate for gradient descent 

        total_steps (int): number of ADAM steps to perform 
        steps_sample_freq (int): how often to sample text from the model / record training/valdiation losses

        key (PRNG key): key for seeding jax random functions

        training_data (str): string of text data to train the model on
        validation_data (str): string of optional validation text data 

        beta1 (float): hyperparameter for ADAM , running gradient average
        beta2 (float): hyperparamter for ADAM, running second moment average

    Returns: 
        history (tuple): losses for the training cycle, sampled with steps_sample_freq
                         
                         step_list - current training step
                         smooth_loss - exponentially smoothed training loss
                         train_loss - training loss 
                         val_loss - loss on validation data 
        params_cache (tuple): cache with all parameters / gradient memories 
        hidden_cache (tuple): cache with final hidden states output 
    '''

    #unique characters in the data set
    chars = set(list(training_data))
    vocab_size = len(chars)

    #character encoding
    char_to_ix = {ch:i for i,ch in enumerate(chars)}
    ix_to_char = {i:ch for i,ch in enumerate(chars)}

    # initialize data generators
    training_generator = get_mini_batch(mini_batch_size,seq_length,char_to_ix,training_data)
    if validation_data:
       validation_generator = get_mini_batch(mini_batch_size,seq_length,char_to_ix,validation_data)

    #current training step counter
    n = 0

    num_layers = len(hidden_sizes)   # number of lstm layers 

    #initialize the model weight matrices

    all_params = {}
    all_grads = {}
    all_sqrd = {}

    h_inputs,c_inputs = {},{}

    h_inputs_val,c_inputs_val={},{} # start val network with blank states each time

    for l in range(num_layers):

        if l==0:
           prev_layer_size = vocab_size
        elif l>0:
           prev_layer_size = hidden_sizes[l-1]

        key,subkey = random.split(key)

        all_params[l],all_grads[l],all_sqrd[l] = initialize_lstm_weights(subkey,hidden_sizes[l],prev_layer_size)

        h_inputs[l] = jnp.zeros((hidden_sizes[l],1))
        c_inputs[l] = jnp.zeros((hidden_sizes[l],1))

        h_inputs_val[l] = jnp.zeros((hidden_sizes[l],1))
        c_inputs_val[l] = jnp.zeros((hidden_sizes[l],1))

    all_params[num_layers],all_grads[num_layers],all_sqrd[num_layers] = initialize_dense_weights(key,hidden_sizes[num_layers-1],vocab_size)

    total_params = 0

    for k1,v1 in all_params.items():
        layer_total = 0
        for k2,v2 in v1.items():
            layer_total += v2.size
            print(k1,k2,v2.shape)

        total_params += layer_total
        print(f'Layer {k1}: {layer_total} parameters')

    print(f'LSTM Model has {num_layers+1} layers with {total_params} parameters')

    #keep list of loss each training step
    step_list = []

    tau = len(training_data) / (mini_batch_size * seq_length)
    alpha = jnp.exp(-4./tau).item()
    #print(f'tau: {tau} alpha: {alpha}')

    smooth_loss = -jnp.log(1.0/vocab_size)*seq_length

    train_losses,smooth_losses,val_losses = [],[],[]

    last_epoch = 0

    sample_size = 200

    sample_ix = [0]*sample_size

    while n < total_steps:
          
          inputs,targets,current_epoch = next(training_generator)

          if current_epoch > last_epoch:

             last_epoch = current_epoch
             
             for l in range(num_layers):
                 h_inputs[l] = jnp.zeros((hidden_sizes[l],1))
                 c_inputs[l] = jnp.zeros((hidden_sizes[l],1))
          
          current_loss, params_cache, hidden_cache = sgd_step_adam(
                                                                current_step=n,
                                                                inputs=inputs,
                                                                targets=targets,
                                                                h_inputs=h_inputs,
                                                                c_inputs=c_inputs,
                                                                all_params=all_params,
                                                                all_grads_mems=all_grads,
                                                                all_sqrd_mems=all_sqrd,
                                                                beta1=beta1,
                                                                beta2=beta2,
                                                                learning_rate=learning_rate)   
          
          smooth_loss = alpha*smooth_loss + (1-alpha)*current_loss

          # need to unpack caches so they are passed to next step
          h_inputs,c_inputs = hidden_cache
          all_params,all_grads,all_sqrd=params_cache

          # sample from the model now and then
          if n % steps_sample_freq == 0:
             key,subkey=random.split(key) #key to split, subkey to gen next random sample

             sample_ix = sample(sample_ix[-1],sample_size,subkey,h_inputs_val,c_inputs_val,all_params)

             txt = ''.join(ix_to_char[int(ix)] for ix in sample_ix)
             txt_wrap = wrap(txt,80)
             txt_wrap = [line.center(100) for line in txt_wrap]
             txt = '\n'.join(txt_wrap)  # \n aren't in the character set so wrap text to make readable

             print('----\n %s \n----' % (txt,))

             print(f'Step n: {n}\t Epoch: {current_epoch}')

             print(f'Train Current: {current_loss:.4f}\tTrain Smoothed: {smooth_loss:.4f}')

             step_list.append(n)
             train_losses.append(current_loss)
             smooth_losses.append(smooth_loss)

             #compute_validation_loss
             if validation_generator:
                inputs_val,targets_val,val_epoch = next(validation_generator)

                val_loss = validation_loss(inputs_val,targets_val,h_inputs_val,c_inputs_val,all_params)

                print(f'Val Current: {val_loss:.4f}') 

                val_losses.append(val_loss)       

          n+=1

    history = (step_list,smooth_losses,train_losses,val_losses)

    return history, params_cache, hidden_cache

def main():

    print('LSTM network test on the alphabet...')

    alphabet_data = 'abcdefghijklmnopqrstuvwxyz0123456789'*2000

    chars = set(list(alphabet_data))

    vocab_size = len(chars)

    char_to_ix = {ch:i for i,ch in enumerate(chars)}
    ix_to_char = {i:ch for i,ch in enumerate(chars)}

    print(f'{vocab_size} unique characters')
    print(f'{len(alphabet_data)} total characters')

    mykey = random.PRNGKey(1)

    seq_length = 10
    hidden_sizes = [8,8]

    num_layers = len(hidden_sizes)

    mini_batch_size = 4

    print(f'Training on sequences with length {seq_length} characters.')

    print(f'Network with {num_layers} LSTM layers sizes {hidden_sizes}')

    total_steps = 4000

    print(f'Will train for {total_steps} steps')

    train_data = alphabet_data[:-500]
    val_data = alphabet_data[-500:]

    history,out_params,out_hidden = train_character_lstm(                  
                                  seq_length=seq_length,
                                  hidden_sizes=hidden_sizes,
                                  mini_batch_size=mini_batch_size,
                                  learning_rate=0.01,
                                  total_steps=total_steps,
                                  steps_sample_freq=250,
                                  key=mykey,
                                  training_data=train_data,
                                  validation_data=val_data,
                                  beta1=0.9,
                                  beta2=0.999)

    step_list = history[0]
    smooth_loss = history[1]
    train_loss = history[2]
    validation_loss = history[3]

    print(f'step \tsmooth loss\t\ttrain loss\t\tval_loss')
    for i in range(len(step_list)):
        print(f'{step_list[i]}\t{smooth_loss[i]}\t{train_loss[i]}\t{validation_loss[i]}')

if __name__=='__main__':
    main()
