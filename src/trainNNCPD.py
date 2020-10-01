#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Neural_Fact import Net, outer_product_np, Fro_Norm, outer_product
from writer import Writer
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
import tensorly as tl
from tensorly import unfold as tl_unfold
from tensorly.decomposition import parafac, non_negative_parafac
from tqdm.notebook import tqdm


# In[2]:


def train(net, X, loss_func, r, epoch = 10, lr = 1e-3, random_init=False):
    '''
    ----- Discription
    
    Training NeuralNNCPD with adam optimization
    ----- Inputs:
    net: A Neural NNCPD object.
    X: The data matrix.
    loss_func: The loss function
    epoch: How many time you want to feed in the data matrix to the network
    '''
    history = Writer()

 
    if(random_init):
        factors_tl = non_negative_parafac(np.asarray(X), rank=r, init='random', random_state=2)
        
    else:
        factors_tl = non_negative_parafac(np.asarray(X), rank=r)
    
    Xs = [Variable(torch.from_numpy(factors_tl[k]), requires_grad=True) for k in range(len(factors_tl))]
    
    
    configs = [{} for i in range(net.depth)]

    for config in configs:
        config['learning_rate'] = lr
        config['beta1'] = 0.9
        config['beta2'] = 0.99
        config['epsilon'] = 1e-8
        config['t'] = 0
            
        
    for i in tqdm(range(epoch)):

        net.zero_grad()
        factors = net(Xs)
        loss = loss_func(net, X, factors)

        loss.backward(retain_graph=True)
        history.add_scalar('loss', loss.data)

        weights = net.weights
        
        for k in range(len(Xs)):
            history.add_tensor('X' + str(k+1), Xs[k].data)
            
        for l in range(net.depth):  
            
            if epoch == 0:
                config['v'] = torch.zeros_like(weights[l].weight.data)
                config['a'] = torch.zeros_like(weights[l].weight.data)
                
            history.add_tensor('S' + str(l+1), weights[l].weight.data)
                
            weights[l].weight.data, configs[l] = adam(weights[l].weight.data, weights[l].weight.grad.data, configs[l])
            weights[l].weight.data = weights[l].weight.data.clamp(min = 0)   
    
    return history

def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('a', torch.zeros_like(w))
    config.setdefault('t', 0)
    
    next_w = None

    v = config['v']
    beta1 = config['beta1']
    beta2 = config['beta2']
    rate = config['learning_rate']
    a = config['a']
    e = config['epsilon']
    t = config['t'] + 1

    nu = 1e-8
    v = beta1 * v + (1 - beta1) * dw

    a = beta2 * a + (1 - beta2) * dw * dw 

    v_c  = v * 1 / (1-beta1**t)
    a_c = a * 1 / (1-beta2**t)


    next_w = w - rate * v_c / (np.sqrt(a_c) + e)


    config['v'] = v
    config['a'] = a
    config['t']  = t

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    return next_w, config
