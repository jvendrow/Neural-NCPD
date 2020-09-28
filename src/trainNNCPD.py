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


# In[2]:


def train_unsupervised(net, X, loss_func, r, epoch = 10, lr1 = 1e-3, lr2 = 1e-3, weight_decay = 1):
    '''
    ----- Discription
    Training the unsupervised Deep_NMF with projection gradient descent
    ----- Inputs:
    net: A Deep_NMF object, note that it should be the unsupervised version, so c = None for the Deep_NMF.
    X: The data matrix.
    loss_func: The loss function, should be a Energy_Loss_func object, with lambd = None or 0
    epoch: How many time you want to feed in the data matrix to the network, default 10
    lr: learning rate, default 1e-3
    weight_decay: the weight decay parameter, doing lr = lr*weight_decay every epoch
    '''
    history = Writer() # creating a Writer object to record the history for the training process

    """
    A,B,C = random_NNCPD(X,r)
    A = Variable(A, requires_grad=True)
    B = Variable(B, requires_grad=True)
    C = Variable(C, requires_grad=True)

    """
 
    factors_tl = non_negative_parafac(np.asarray(X), rank=r)#, init='random', random_state=2)
    print("finished tensorly")
    
    #factors_tl = non_negative_parafac(np.asarray(X), rank=r) #kill 9 error?

    #side_X = X.shape[0]
    #factor = 0
    #A = Variable(torch.from_numpy(factors_tl[0] + factor * np.abs(np.random.randn(side_X, r))), requires_grad=True)
    #B = Variable(torch.from_numpy(factors_tl[1] + factor * np.abs(np.random.randn(side_X, r))), requires_grad=True)
    #C = Variable(torch.from_numpy(factors_tl[2] + factor * np.abs(np.random.randn(side_X, r))), requires_grad=True)
    A = Variable(torch.from_numpy(factors_tl[0]), requires_grad=True)
    B = Variable(torch.from_numpy(factors_tl[1]), requires_grad=True)
    C = Variable(torch.from_numpy(factors_tl[2]), requires_grad=True)
    
    """
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(4,5))
    color = "viridis"
    axs[0].imshow(A.detach().numpy(), cmap=plt.get_cmap(color))
    axs[1].imshow(B.detach().numpy(), cmap=plt.get_cmap(color))
    axs[2].imshow(C.detach().numpy(), cmap=plt.get_cmap(color))
    plt.show()
    print(np.max(A.detach().numpy()))
    print(np.max(B.detach().numpy()))
    print(np.max(C.detach().numpy()))

    X_approx = outer_product_np(A.detach().numpy(), B.detach().numpy(), C.detach().numpy())
    print("X - ABC Approximation")
    print(np.linalg.norm(np.ndarray.flatten(np.asarray(X)-X_approx), 2))
    
    X_max = np.max(X_approx,axis=0)
    fig = plt.figure(figsize = (3,3))
    plt.imshow(X_max)
    plt.show()
    """

    
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
        
        
    for i in range(epoch):
        
        #X_approx = outer_product_np(A.detach().numpy(), B.detach().numpy(), C.detach().numpy())
        #print(np.linalg.norm(np.ndarray.flatten(np.asarray(X)-X_approx), 2))

        net.zero_grad()
        A_S_lst, B_S_lst, C_S_lst = net(A,B,C)
        loss = loss_func(net, X, A_S_lst, B_S_lst, C_S_lst, A, B, C)
        #A.requires_grad_()
        #B.requires_grad_()
        #C.requires_grad_()
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
                #projection gradient descent
                #As[j].data = As[j].data.sub_(lr2*As[j].grad.data)
                As[j].data, configs[l][j] = adam(As[j].data, As[j].grad.data, configs[l][j])
                As[j].data = As[j].data.clamp(min = 0)   
            
            #if l % 3 == 0:
            #A.data = A.data.sub_(lr1*A.grad.data)
            A.data, ABC_configs[0] = adam(A.data, A.grad.data, ABC_configs[0])
            A.data = A.data.clamp(min = 0)  
            #if l % 3 == 1:    
            #B.data = B.data.sub_(lr1*B.grad.data)
            B.data, ABC_configs[1] = adam(B.data, B.grad.data, ABC_configs[1])
            B.data = B.data.clamp(min = 0)  
            #if l % 3 == 2:
            #C.data = C.data.sub_(lr1*C.grad.data)
            C.data, ABC_configs[2] = adam(C.data, C.grad.data, ABC_configs[2])
            C.data = C.data.clamp(min = 0)  
            
        lr1 = lr1*weight_decay
        lr2 = lr2*weight_decay
        #print(A.grad)
        if (i)%10 == 0:
            print('epoch = ', i+1, '\n', loss.data)
            
    
    X_approx = outer_product_np(A.detach().numpy(), B.detach().numpy(), C.detach().numpy())
    print("X - ABC Approximation")
    print(np.linalg.norm(np.ndarray.flatten(np.asarray(X)-X_approx), 2))
    
    """
    X_max = np.max(X_approx,axis=0)
    fig = plt.figure(figsize = (3,3))
    plt.imshow(X_max)
    plt.show()
    """
    """
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(9,9))
    color = "viridis"
    #color = "binary"
    X_max = np.max(X_approx,axis=0)
    fig = plt.figure(figsize = (3,3))
    #plt.imshow(X_max)
    axs[0].imshow(X_max,cmap=plt.get_cmap(color))
    #plt.show()
    X_max = np.max(X_approx,axis=1)
    fig = plt.figure(figsize = (3,3))
    #plt.imshow(X_max)
    axs[1].imshow(X_max,cmap=plt.get_cmap(color))
    #plt.show()
    X_max = np.max(X_approx,axis=2)
    fig = plt.figure(figsize = (3,3))
    #plt.imshow(X_max)
    axs[2].imshow(X_max,cmap=plt.get_cmap(color))
    plt.show()
    
    A_X_approx = A_S_lst[-1]
    B_X_approx = B_S_lst[-1]
    C_X_approx = C_S_lst[-1]
    for i in range(net.depth-2, -1, -1):
        A_X_approx = torch.mm(net.A_lsqnonneglst[i].A,A_X_approx)
        B_X_approx = torch.mm(net.B_lsqnonneglst[i].A,B_X_approx)
        C_X_approx = torch.mm(net.C_lsqnonneglst[i].A,C_X_approx)
    
    
    X_approx = outer_product(A_X_approx,B_X_approx,C_X_approx)
    print("AS approximation:")
    print(np.linalg.norm(np.ndarray.flatten(np.asarray(X)-X_approx.detach().numpy()), 2))
    
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(9,9))

    color = "viridis"
    #color = "binary"
    X_max = np.max(X_approx.detach().numpy(),axis=0)
    fig = plt.figure(figsize = (3,3))
    #plt.imshow(X_max)
    axs[0].imshow(X_max,cmap=plt.get_cmap(color))
    #plt.show()
    X_max = np.max(X_approx.detach().numpy(),axis=1)
    fig = plt.figure(figsize = (3,3))
    #plt.imshow(X_max)
    axs[1].imshow(X_max,cmap=plt.get_cmap(color))
    #plt.show()
    X_max = np.max(X_approx.detach().numpy(),axis=2)
    fig = plt.figure(figsize = (3,3))
    #plt.imshow(X_max)
    axs[2].imshow(X_max,cmap=plt.get_cmap(color))
    plt.show()
    
    X = X_approx.detach().numpy()
    fig, axs = plt.subplots(2, 5, constrained_layout=True, figsize=(12,5))
    color = "viridis"
    for i in range(5):
        axs[0][i].imshow(X[:,:,i], cmap=plt.get_cmap(color))
        axs[1][i].imshow(X[:,:,i+5], cmap=plt.get_cmap(color))
    plt.show()
    """
    
        
    return history


# In[ ]:


def train_supervised(net, X, loss_func, label, L= None, epoch = 10, lr_nmf = 1e-3, lr_classification = 1e-3, weight_decay = 1):
    '''
    ---- Description
    Training the supervised Deep_NMF with projection gradient descent. Details for the training process:
        for each epoch we update the NMF layer and the classification layer separately. First update the NMF layer for once
        and then update the classification layer for thirty times. The learning rate is 
    ---- Inputs:
    net: A Deep_NMF object, note that it should be the unsupervised version, so c = None for the Deep_NMF.
    X: The data matrix.
    epoch: How many time you want to feed in the data matrix to the network, default 10
    loss_func: The loss function, should be a Energy_Loss_func object
    epoch: How many time you want to feed in the data matrix to the network, default 10
    lr_nmf: the learning rate for the NMF layer
    lr_classification: the learning rate for the classification layer
    weight_decay: the weight decay parameter, doing lr = lr*weight_decay every epoch
    '''
    
    history = Writer() # creating a Writer object to record the history for the training process
    for i in range(epoch):
        
        # doing gradient update for NMF layer
        net.zero_grad()
        S_lst, pred = net(X)
        loss = loss_func(net, X, S_lst, pred, label, L)
        loss.backward()
        for l in range(net.depth - 1):
            history.add_scalar('loss',loss.data)
            A = net.lsqnonneglst[l].A
            # record history
            history.add_tensor('A'+str(l+1), A.data)
            history.add_tensor('grad_A'+str(l+1), A.grad.data)
            history.add_tensor('S' + str(l+1), S_lst[l].data)
            # projection gradient descent
            A.data = A.data.sub_(lr_nmf*A.grad.data)
            A.data = A.data.clamp(min = 0)
            
        # doing gradient update for classification layer
        for iter_classifier in range(30):
            net.zero_grad()
            S_lst, pred = net(X)
            loss = loss_func(net, X, S_lst, pred, label, L)
            loss.backward()
            S_lst[0].detach()
            history.add_scalar('loss',loss.data)
            weight = net.linear.weight
            weight.data = weight.data.sub_(lr_classification*weight.grad.data)
            history.add_tensor('weight', weight.data.clone())
            history.add_tensor('grad_weight', weight.grad.data.clone())
        
        
        lr_nmf = lr_nmf*weight_decay
        lr_classification = lr_classification*weight_decay
        
        print('epoch = ', i+1, '\n', loss.data)
    return history


# In[ ]:


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

