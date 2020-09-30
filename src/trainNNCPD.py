#!/usr/bin/env python
# coding: utf-8

# In[1]:


from NNCPD import NNCPD, Recon_Loss, L21_Norm, random_NNCPD, outer_product_np, Fro_Norm, outer_product
from writer import Writer
import torch
import numpy as np
from lsqnonneg_module import LsqNonneg
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
import tensorly as tl
from tensorly import unfold as tl_unfold
from tensorly.decomposition import parafac, non_negative_parafac
from tqdm.notebook import tqdm


# In[2]:


def train(net, X, loss_func, r, epoch = 10, lr1 = 1e-3, lr2 = 1e-3, random_init=False):
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
        factors_tl = non_negative_parafac(np.asarray(X), rank=r, init='random', random_state=2)[1]
        
    else:
        factors_tl = non_negative_parafac(np.asarray(X), rank=r)[1]
    
    A = Variable(torch.from_numpy(factors_tl[0]), requires_grad=True)
    B = Variable(torch.from_numpy(factors_tl[1]), requires_grad=True)
    C = Variable(torch.from_numpy(factors_tl[2]), requires_grad=True)
    
    
    configs = [[{} for _ in range(3)] for i in range(net.depth-1)]

    for config_list in configs:
        for config in config_list:
            config['learning_rate'] = lr2
            config['beta1'] = 0.9
            config['beta2'] = 0.99
            config['epsilon'] = 1e-8
            config['t'] = 0
            
    ABC_configs = [{} for _ in range(3)]

    for config in ABC_configs:
        config['learning_rate'] = lr1
        config['beta1'] = 0.9
        config['beta2'] = 0.99
        config['epsilon'] = 1e-8
        config['t'] = 0
        
        
    for i in tqdm(range(epoch)):
        


        net.zero_grad()
        A_S_lst, B_S_lst, C_S_lst = net(A,B,C)
        loss = loss_func(net, X, A_S_lst, B_S_lst, C_S_lst, A, B, C)

        loss.backward(retain_graph=True)
        history.add_scalar('loss', loss.data)

        for l in range(net.depth - 1):
            As = [net.A_lsqnonneglst[l].A, net.B_lsqnonneglst[l].A,net.C_lsqnonneglst[l].A]
            Ss = [A_S_lst, B_S_lst, C_S_lst]
            Xs = [A, B, C]
            
            if epoch == 0:
                for j, config in enumerate(configs[l]):
                    config['v'] = torch.zeros_like(As[j].data)
                    config['a'] = torch.zeros_like(As[j].data)
                
            
            for j, letter in enumerate(['A','B','C']):
                # record history
                history.add_tensor(letter + '_A'+str(l+1), As[j].data)
                history.add_tensor(letter + '_grad_A'+str(l+1), As[j].grad.data)
                history.add_tensor(letter + '_S' + str(l+1), Ss[j][l].data)
                history.add_tensor(letter + '_X' + str(l+1), Xs[j].data)
                
                As[j].data, configs[l][j] = adam(As[j].data, As[j].grad.data, configs[l][j])
                As[j].data = As[j].data.clamp(min = 0)   

            A.data, ABC_configs[0] = adam(A.data, A.grad.data, ABC_configs[0])
            A.data = A.data.clamp(min = 0)  

            B.data, ABC_configs[1] = adam(B.data, B.grad.data, ABC_configs[1])
            B.data = B.data.clamp(min = 0)  

            C.data, ABC_configs[2] = adam(C.data, C.grad.data, ABC_configs[2])
            C.data = C.data.clamp(min = 0)  
 
        
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
