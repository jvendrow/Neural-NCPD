#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Author: Josh Vendrow


# In[2]:


import torch
import torch.nn as nn
from torch.autograd import Variable
from matplotlib import pyplot as plt
import scipy.io as sio
from lsqnonneg_module import LsqNonneg
import numpy as np
import torch.nn.functional as F


# In[3]:


class NNCPD(nn.Module):
    '''
    Build an NNCPD network structure.
    
    initial parameters:
    depth_info: list, [m, k1, k2,...k_L] # Note! m must be contained in the list, which is different from the matlab version
    c: default None, otherwise it should be a scalar indicating how many classes there are
    
    the Deep_NMF object contains several NMF layers(contained in self.lsqnonneglst, each element in the list self.lsqnonneglst is a Lsqnonneg object)
    and a linear layer for classification(self.linear).
    '''
    def __init__(self, A_depth_info, B_depth_info, C_depth_info, c = None):
        super(NNCPD, self).__init__()
        self.A_depth_info = A_depth_info
        self.B_depth_info = B_depth_info
        self.C_depth_info = C_depth_info
        self.depth = len(A_depth_info)
        self.c= c
        self.A_lsqnonneglst = nn.ModuleList([LsqNonneg(A_depth_info[i], A_depth_info[i+1]) 
                                           for i in range(self.depth-1)])
        self.B_lsqnonneglst = nn.ModuleList([LsqNonneg(B_depth_info[i], B_depth_info[i+1]) 
                                           for i in range(self.depth-1)])
        self.C_lsqnonneglst = nn.ModuleList([LsqNonneg(C_depth_info[i], C_depth_info[i+1]) 
                                           for i in range(self.depth-1)])
        if c is not None:
            self.linear = nn.Linear(A_depth_info[-1],c, bias = False).double()
    
    def forward(self, A, B, C):
        A_S_lst = []
        B_S_lst = []
        C_S_lst = []
        
        if(self.depth < 1):
            return
        
        A_new = self.A_lsqnonneglst[0](A)
        A_S_lst.append(A_new)

        B_new = self.B_lsqnonneglst[0](B)
        B_S_lst.append(B_new)

        C_new = self.C_lsqnonneglst[0](C)
        C_S_lst.append(C_new)
           
        for i in range(1,self.depth-1):
            A_new = self.A_lsqnonneglst[i](A_new)
            A_S_lst.append(A_new)

            B_new = self.B_lsqnonneglst[i](B_new)
            B_S_lst.append(B_new)

            C_new = self.C_lsqnonneglst[i](C_new)
            C_S_lst.append(C_new)
            
        if self.c is None:
            return A_S_lst, B_S_lst, C_S_lst
        #else:
        #    pred = self.linear(X)
        #    return S_lst, pred
        


# In[4]:


def random_NNCPD(X, r):
    
    n1, n2, n3 = X.shape
    A = torch.autograd.Variable(torch.abs(torch.randn(n1, r, requires_grad=True)))
    B = torch.autograd.Variable(torch.abs(torch.randn(n2, r, requires_grad=True)))
    C = torch.autograd.Variable(torch.abs(torch.randn(n3, r, requires_grad=True)))
    return A, B, C


# In[ ]:


def PTF(X, r):
    
    threshold = 0.01
    
    n1, n2, n3 = X.shape
    
    A = np.absolute(np.random.randn(n1, r))
    B = np.absolute(np.random.randn(n2, r))
    C = np.absolute(np.random.randn(n3, r))
    
    
    prev_loss = nn.MSELoss(X, outer_product_np(A,B,C))
    
    
    while(true):

        Y = np.transpose(X, (0,1,2))
        Y = np.reshape(Y, (n1,n2*n3))
        #M = np.reshape(outer_product2(B,C), (n2*n3,)
        #M = np.
        
        divA = (Y*M)/(A*M.T*M)
        
        
        
        
        Y = np.transpose(X, (1,0,2))
        Y = np.reshape((n2,n1*n3))
        M = outer_product2(A,C)
        
        divB = (Y*M)/(B*M.T*M)
        
        
        
        Y = np.transpose(X, (2,0,1))
        Y = np.reshape((n3,n1*n2))
        M = outer_product2(A,B)
        
        divC = (Y*M)/(C*M.T*M)
        
        A = np.multiply(A, divA)
        B = np.multiply(B, divB)  
        C = np.multiply(A, divC)
        loss = nn.MSELoss(X, outer_product_np(A,B,C))
        if(prev_loss - loss < threshold):
            break
             
    
    return A, B, C


# In[ ]:


def weights_H(A, B, C, H_A, H_B, H_C):
    
    alphas = []
    n_1, r = A.shape
    n_2, r = B.shape
    n_3, r = C.shape
    weights = torch.zeros((r, r, 2))
    
    
    for col in range(r):
        for it in range(r**2):
            i = int(it % r)
            j = int((it/r) % r)

            weights[col, i, 0] += torch.sum(H_A[i] * H_B[j] * H_C[col]);
            weights[col, j, 1] += torch.sum(H_A[i] * H_B[j] * H_C[col]);
         


    return weights


# In[ ]:


def intro_product(A, B, C, H_A, H_B, H_C, get_alpha=False):
    
    alphas = []
    n_1, r = A.shape
    n_2, r = B.shape
    n_3, r = C.shape
    X_approx = torch.zeros((n_1,n_2,n_3))
    
    
    for it in range(r**3):
        i = int(it % r)
        j = int((it/r) % r)
        k = int((it/(r**2) % r))
        
        alpha = torch.sum(H_A[i] * H_B[j] * H_C[k]);
        alphas.append(alpha)
        X_approx += alpha * torch.einsum('p, qr->pqr',A[:,i],torch.ger(B[:,j],C[:,k]))
         
    if get_alpha:
        return X_approx, alphas
    else:
         return X_approx
        
        
        


# In[ ]:


def outer_product(A, B, C):
    
    """
    Calculates the outer product ABC = sum_{i=1}^r a_i (outer) b_i (outer) c_i
    """
    n_1, r = A.shape
    n_2, r = B.shape
    n_3, r = C.shape
    X_approx = torch.zeros((n_1,n_2,n_3))
    
    for i in range(r):
        temp = torch.einsum('p, qr->pqr',A[:,i],torch.ger(B[:,i],C[:,i]))
        X_approx += temp
        
    return X_approx

def outer_product_np(A, B, C):
    
    """
    Calculates the outer product ABC = sum_{i=1}^r a_i (outer) b_i (outer) c_i
    """
    n_1, r = A.shape
    n_2, r = B.shape
    n_3, r = C.shape
    X_approx = np.zeros((n_1,n_2,n_3))
    
    for i in range(r):
        temp = np.multiply.outer(A[:,i],np.outer(B[:,i],C[:,i]))
        X_approx += temp
        
    return X_approx

def outer_product2(A, B):
    
    """
    Calculates the outer product AB = sum_{i=1}^r a_i (outer) b_i
    """
    n_1, r = A.shape
    n_2, r = B.shape
    X_approx = np.zeros((n_1,n_2))
    
    for i in range(r):
        X_approx += np.outer(A[:,i],B[:,i])
        
    return X_approx

class Recon_Loss(nn.Module):
    '''
    Defining the reconstruction loss || X - ABC ||
    
    initial parameter: 
        lambd: the regularization parameter, defining how important the classification error is.
        classification_type: string, 'L2' or 'CrossEntropy'. Default 'CrossEntropy'
    '''
    def __init__(self,lambd = 0, classification_type = 'CrossEntropy'):
        super(Recon_Loss, self).__init__()
        
        self.criterion = Fro_Norm()

            
    def forward(self, net, X, A_S_lst, B_S_lst, C_S_lst, pred = None, label = None, L = None):
        depth = net.depth
        
        A_X_approx = A_S_lst[-1]
        B_X_approx = B_S_lst[-1]
        C_X_approx = C_S_lst[-1]
        for i in range(depth-2, -1, -1):
            A_X_approx = torch.mm(net.A_lsqnonneglst[i].A,A_X_approx)
            B_X_approx = torch.mm(net.B_lsqnonneglst[i].A,B_X_approx)
            C_X_approx = torch.mm(net.C_lsqnonneglst[i].A,C_X_approx)
            
        X_approx = outer_product(A_X_approx,B_X_approx,C_X_approx)
        
        reconstructionloss = self.criterion(X_approx, X)

        if pred is None:
            # unsupervised case
            assert(label is None and L is None)
            return reconstructionloss
        else:
            # fully supervised case and semisupervised case
            #classificationloss = self.criterion2(pred, label, L)
            #return reconstructionloss + self.lambd*classificationloss
            return reconstructionloss
        
class Recon_Loss_Straight(nn.Module):
    '''
    Defining the reconstruction loss || X - ABC ||
    
    initial parameter: 
        lambd: the regularization parameter, defining how important the classification error is.
        classification_type: string, 'L2' or 'CrossEntropy'. Default 'CrossEntropy'
    '''
    def __init__(self,lambd = 0, classification_type = 'CrossEntropy'):
        super(Recon_Loss_Straight, self).__init__()
        
        self.criterion = Fro_Norm()
        self.l1 = nn.L1Loss(size_average=False)


            
    def forward(self, net, X, A_S_lst, B_S_lst, C_S_lst, A, B, C, pred = None, label = None, L = None):
        depth = net.depth
        
        #l1_norms = self.l1(A_S_lst[-1],target=torch.zeros_like(A_S_lst[-1]))
        #l1_norms += self.l1(B_S_lst[-1],target=torch.zeros_like(B_S_lst[-1]))
        #l1_norms += self.l1(C_S_lst[-1],target=torch.zeros_like(C_S_lst[-1]))
        
        A_X_approx = A_S_lst[-1]
        B_X_approx = B_S_lst[-1]
        C_X_approx = C_S_lst[-1]
        for i in range(depth-2, -1, -1):
            A_X_approx = torch.mm(net.A_lsqnonneglst[i].A,A_X_approx)
            B_X_approx = torch.mm(net.B_lsqnonneglst[i].A,B_X_approx)
            C_X_approx = torch.mm(net.C_lsqnonneglst[i].A,C_X_approx)
            
            #l1_norms += self.l1(net.A_lsqnonneglst[i].A,target=torch.zeros_like(net.A_lsqnonneglst[i].A))
            #l1_norms += self.l1(net.B_lsqnonneglst[i].A,target=torch.zeros_like(net.B_lsqnonneglst[i].A))
            #l1_norms += self.l1(net.C_lsqnonneglst[i].A,target=torch.zeros_like(net.C_lsqnonneglst[i].A))
            
        A_loss = self.criterion(A, A_X_approx)
        B_loss = self.criterion(B, B_X_approx)
        C_loss = self.criterion(C, C_X_approx)
            
        X_approx = outer_product(A_X_approx,B_X_approx,C_X_approx)
        
        X_approx2 = outer_product(A,B,C)
        
        
        reconstructionloss = self.criterion(X_approx, X)# + 0.00003 * l1_norms
        reconstructionloss += self.criterion(X_approx2, X)
        
        #reconstructionloss += A_loss + B_loss + C_loss
        if pred is None:
            # unsupervised case
            assert(label is None and L is None)
            return reconstructionloss
        else:
            # fully supervised case and semisupervised case
            classificationloss = self.criterion2(pred, label, L)
            return reconstructionloss + self.lambd*classificationloss
        
class Recon_Loss_NMF(nn.Module):
    '''
    Defining the reconstruction loss || X - ABC ||
    
    initial parameter: 
        lambd: the regularization parameter, defining how important the classification error is.
        classification_type: string, 'L2' or 'CrossEntropy'. Default 'CrossEntropy'
    '''
    def __init__(self,lambd = 0, classification_type = 'CrossEntropy'):
        super(Recon_Loss_NMF, self).__init__()
        
        self.criterion = Fro_Norm()
        self.l1 = nn.L1Loss(size_average=False)


            
    def forward(self, net, X, A_S_lst, B_S_lst, C_S_lst, A, B, C, pred = None, label = None, L = None):
        depth = net.depth
        
        #l1_norms = self.l1(A_S_lst[-1],target=torch.zeros_like(A_S_lst[-1]))
        #l1_norms += self.l1(B_S_lst[-1],target=torch.zeros_like(B_S_lst[-1]))
        #l1_norms += self.l1(C_S_lst[-1],target=torch.zeros_like(C_S_lst[-1]))
        
        A_X_approx = A_S_lst[-1]
        B_X_approx = B_S_lst[-1]
        C_X_approx = C_S_lst[-1]
        for i in range(depth-2, -1, -1):
            A_X_approx = torch.mm(net.A_lsqnonneglst[i].A,A_X_approx)
            B_X_approx = torch.mm(net.B_lsqnonneglst[i].A,B_X_approx)
            C_X_approx = torch.mm(net.C_lsqnonneglst[i].A,C_X_approx)
            
            #l1_norms += self.l1(net.A_lsqnonneglst[i].A,target=torch.zeros_like(net.A_lsqnonneglst[i].A))
            #l1_norms += self.l1(net.B_lsqnonneglst[i].A,target=torch.zeros_like(net.B_lsqnonneglst[i].A))
            #l1_norms += self.l1(net.C_lsqnonneglst[i].A,target=torch.zeros_like(net.C_lsqnonneglst[i].A))
            
        A_loss = self.criterion(A, A_X_approx)
        B_loss = self.criterion(B, B_X_approx)
        C_loss = self.criterion(C, C_X_approx)
            
        X_approx = outer_product(A_X_approx,B_X_approx,C_X_approx)
        
        X_approx2 = outer_product(A,B,C)
        
        
        #reconstructionloss = self.criterion(X_approx, X)# + 0.00003 * l1_norms
        reconstructionloss = self.criterion(X_approx2, X)
        
        reconstructionloss = reconstructionloss + A_loss + B_loss + C_loss
        if pred is None:
            # unsupervised case
            assert(label is None and L is None)
            return reconstructionloss
        else:
            # fully supervised case and semisupervised case
            classificationloss = self.criterion2(pred, label, L)
            return reconstructionloss + self.lambd*classificationloss
        
class Energy_Loss_NNCPD(nn.Module):
    '''
    Defining the reconstruction loss || X - ABC ||
    
    initial parameter: 
        lambd: the regularization parameter, defining how important the classification error is.
        classification_type: string, 'L2' or 'CrossEntropy'. Default 'CrossEntropy'
    '''
    def __init__(self,lambd = 0, classification_type = 'CrossEntropy'):
        super(Energy_Loss_NNCPD, self).__init__()
        
        self.criterion = Fro_Norm()
        self.criterion2 = Energy_Loss_NMF()


            
    def forward(self, net, X, A_S_lst, B_S_lst, C_S_lst, A, B, C, pred = None, label = None, L = None):
        depth = net.depth
        
        
        A_X_approx = A_S_lst[-1]
        B_X_approx = B_S_lst[-1]
        C_X_approx = C_S_lst[-1]
        for i in range(depth-2, -1, -1):
            A_X_approx = torch.mm(net.A_lsqnonneglst[i].A,A_X_approx)
            B_X_approx = torch.mm(net.B_lsqnonneglst[i].A,B_X_approx)
            C_X_approx = torch.mm(net.C_lsqnonneglst[i].A,C_X_approx)
            
            
        A_loss = self.criterion2(A, A_S_lst, net.A_lsqnonneglst)
        B_loss = self.criterion2(B, B_S_lst, net.B_lsqnonneglst)
        C_loss = self.criterion2(C, C_S_lst, net.C_lsqnonneglst)
            
        X_approx = outer_product(A_X_approx,B_X_approx,C_X_approx)
        
        X_approx2 = outer_product(A,B,C)
        
        
        reconstructionloss = self.criterion(X_approx, X)# + 0.00003 * l1_norms
        #reconstructionloss = 0#self.criterion(X_approx2, X)
        
        reconstructionloss = reconstructionloss + A_loss + B_loss + C_loss
        if pred is None:
            # unsupervised case
            assert(label is None and L is None)
            return reconstructionloss
        else:
            # fully supervised case and semisupervised case
            classificationloss = self.criterion2(pred, label, L)
            return reconstructionloss + self.lambd*classificationloss


# In[ ]:


## Defining all kinds of loss functions that is needed

class Fro_Norm(nn.Module):
    '''
    calculate the Frobenius norm between two matrices of the same size.
    Do: criterion = Fro_Norm()
        loss = criterion(X1,X2) and the loss is the entrywise average of the square of Frobenius norm.
    '''
    def __init__(self):
        super(Fro_Norm, self).__init__()
        self.criterion = nn.MSELoss()
    def forward(self,X1, X2):
        len1 = torch.numel(X1.data)
        len2 = torch.numel(X2.data)
        assert(len1 == len2)
        X = X1 - X2
        #X.contiguous()
        #return self.criterion(X.view(len1), Variable(torch.zeros(len1).double()))
        return self.criterion(X.reshape((len1)), Variable(torch.zeros(len1).double()))

    
class ReconstructionLoss(nn.Module):
    '''
    calculate the reconstruction error ||X - ABC||_F^2.
    Do: criterion = ReconstructionLoss()
        loss = criterion(X, S, A) and the loss is the entrywise average of the square of Frobenius norm ||X - AS||_F^2.
    '''
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.criterion = Fro_Norm()
        
    def forward(self, X, A, B, C):
        X_approx = outer_product(A, B, C)
        reconstructionloss = self.criterion(X_approx, X)
        return reconstructionloss
   
"""
class ReconstructionLoss(nn.Module):
    '''
    calculate the reconstruction error ||X - AS||_F^2.
    Do: criterion = ReconstructionLoss()
        loss = criterion(X, S, A) and the loss is the entrywise average of the square of Frobenius norm ||X - AS||_F^2.
    '''
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.criterion = Fro_Norm()
    def forward(self, X, S, A):
        X_approx = torch.mm(A,S)
        reconstructionloss = self.criterion(X_approx, X)
        return reconstructionloss
"""

class Energy_Loss_NMF(nn.Module):
    """
    Defining the energy loss function as in the Neural NMF Paper. #Jamie: can we add math description?
    ...
    Parameters
    ----------
    lambd: float, optional
        The regularization parameter, defining weight of classification error in loss function. 
    classification_type: string, optional
        Classification loss indicator 'L2' or 'CrossEntropy' (default 'CrossEntropy').
    
    Methods
    ----------
    forward(net,X,S_lst)
        Forward propagates and computes energy loss value.
    """
    def __init__(self,lambd = 0, classification_type = 'CrossEntropy'):
        super(Energy_Loss_NMF, self).__init__()
        self.lambd = lambd
        self.classification_type = classification_type
        self.criterion1 = ReconstructionLossNMF()
        if classification_type == 'L2':
            self.criterion2 = ClassificationLossL2()
        else:
            self.criterion2 = ClassificationLossCrossEntropy()
            
    def forward(self, X, S_lst, lsqnonneglst, pred = None, label = None, L = None):
        """
        Runs the forward pass of the energy loss function.
        Parameters
        ----------
        net: Pytorch module Neural NMF object
            The Neural NMF object for which the loss is calculated.
        X: Pytorch tensor
            The input to the Neural NMF network (matrix to be factorized).
        S_lst: list
            All S matrices ([S_0, S_1, ..., S_L]) that were returned by the forward pass of the Neural 
            NMF object.
        pred: Pytorch tensor, optional
            The approximation to the classification one-hot indicator matrix of size c x n produced
            by forward pass (B*S_L) (default is None).
        label: Pytorch tensor, optional
            The classification (label) matrix for supervised model.  If the classification_type is 'L2',
            this matrix is a one-hot encoding matrix of size c x n.  If the classification_type is
            'CrossEntropy', this matrix is of size n with elements in [0,c-1] (default is None).
        L: Pytorch tensor, optional
            The label indicator matrix for semi-supervised model that indicates if labels are known 
            for n data points, of size c x n with columns of all ones or all zeros to indicate if label
            for that data point is known (default is None).
        Returns
        -------
        reconstructionloss: Pytorch tensor
            The total energy loss from X, the S matrices, and the A matrices, stored in a 1x1 Pytorch 
            tensor to preserve information for backpropagation.
        """

        total_reconstructionloss = self.criterion1(X, S_lst[0], lsqnonneglst[0].A)

        depth = len(S_lst)+1
        for i in range(1,depth-1):
            total_reconstructionloss += self.criterion1(S_lst[i-1], S_lst[i], lsqnonneglst[i].A)
        if pred is None:
            # unsupervised case
            assert(label is None and L is None)
            return total_reconstructionloss
        else:
            # fully supervised case and semisupervised case
            classificationloss = self.criterion2(pred, label, L)
            return total_reconstructionloss + self.lambd*classificationloss


# In[1]:


class Energy_Loss_Tensor(nn.Module):
    '''
    Defining the reconstruction loss || X - ABC ||
    
    initial parameter: 
        lambd: the regularization parameter, defining how important the classification error is.
        classification_type: string, 'L2' or 'CrossEntropy'. Default 'CrossEntropy'
    '''
    def __init__(self,lambd = 0, l1_reg = 0, classification_type = 'CrossEntropy'):
        super(Energy_Loss_Tensor, self).__init__()
        
        self.criterion = Fro_Norm()
        self.l1 = nn.L1Loss(size_average=False)
        self.l1_reg = l1_reg

            
    def forward(self, net, X, A_S_lst, B_S_lst, C_S_lst, A, B, C, pred = None, label = None, L = None):
        depth = net.depth
        
        
        approx = []
        #print(depth)
        #print(len(A_S_lst))
        
        for d in range(depth-1):
        #for d in range(0,2):   
            A_X_approx = A_S_lst[d]
            B_X_approx = B_S_lst[d]
            C_X_approx = C_S_lst[d]
            for i in range(d, -1, -1):
                A_X_approx = torch.mm(net.A_lsqnonneglst[i].A,A_X_approx)
                B_X_approx = torch.mm(net.B_lsqnonneglst[i].A,B_X_approx)
                C_X_approx = torch.mm(net.C_lsqnonneglst[i].A,C_X_approx)
                
            approx.append(outer_product(A_X_approx,B_X_approx,C_X_approx))

        X_approx = outer_product(A,B,C)
        #reconstructionloss = self.criterion(X, X_approx)
        #reconstructionloss = self.criterion(X_approx, approx[0])
        #for i in range(depth-2):
        #    reconstructionloss += self.criterion(approx[i], approx[i+1])
        reconstructionloss = self.criterion(X, approx[0])
        for i in range(1,depth-1):
            reconstructionloss += self.criterion(X, approx[i])
        

        for d in range(depth-1):
            reconstructionloss += self.l1_reg * torch.norm(A_S_lst[d], 1)
            reconstructionloss += self.l1_reg * torch.norm(B_S_lst[d], 1)
            reconstructionloss += self.l1_reg * torch.norm(C_S_lst[d], 1)
        """
        for d in range(depth-1):
            reconstructionloss += self.l1_reg * self.l1(A_S_lst[d],target=torch.zeros_like(A_S_lst[d]))
            reconstructionloss += self.l1_reg * self.l1(B_S_lst[d],target=torch.zeros_like(B_S_lst[d]))
            reconstructionloss += self.l1_reg * self.l1(C_S_lst[d],target=torch.zeros_like(C_S_lst[d]))
         """
            
        if pred is None:
            # unsupervised case
            assert(label is None and L is None)
            return reconstructionloss
        else:
            # fully supervised case and semisupervised case
            #classificationloss = self.criterion2(pred, label, L)
            #return reconstructionloss + self.lambd*classificationloss
            return reconstructionloss


# In[ ]:


class Energy_Loss_Tensor2(nn.Module):
    '''
    Defining the reconstruction loss || X - ABC ||
    
    initial parameter: 
        lambd: the regularization parameter, defining how important the classification error is.
        classification_type: string, 'L2' or 'CrossEntropy'. Default 'CrossEntropy'
    '''
    def __init__(self,lambd = 0, classification_type = 'CrossEntropy'):
        super(Energy_Loss_Tensor2, self).__init__()
        
        self.criterion = Fro_Norm()

            
    def forward(self, net, X, A_S_lst, B_S_lst, C_S_lst, A, B, C, pred = None, label = None, L = None):
        depth = net.depth
        
        
        approx = []
        #print(depth)
        #print(len(A_S_lst))
        
        for d in range(depth-1):
        #for d in range(0,2):   
            A_X_approx = net.A_lsqnonneglst[d].A
            B_X_approx = net.B_lsqnonneglst[d].A
            C_X_approx = net.C_lsqnonneglst[d].A
            for i in range(d-1, -1, -1):
                A_X_approx = torch.mm(net.A_lsqnonneglst[i].A,A_X_approx)
                B_X_approx = torch.mm(net.B_lsqnonneglst[i].A,B_X_approx)
                C_X_approx = torch.mm(net.C_lsqnonneglst[i].A,C_X_approx)
                
            approx.append(outer_product(A_X_approx,B_X_approx,C_X_approx))

        X_approx = outer_product(A,B,C)
        reconstructionloss = self.criterion(X, X_approx)
        #reconstructionloss += self.criterion(X_approx, approx[0])
        for i in range(depth-1):# -2 !
            reconstructionloss += self.criterion(approx[i], X)#approx[i+1])
        
        if pred is None:
            # unsupervised case
            assert(label is None and L is None)
            return reconstructionloss
        else:
            # fully supervised case and semisupervised case
            #classificationloss = self.criterion2(pred, label, L)
            #return reconstructionloss + self.lambd*classificationloss
            return reconstructionloss


# In[ ]:


class ReconstructionLossNMF(nn.Module):
    """
    Calculates the entrywise average of the square of Frobenius norm ||X - AS||_F^2.
    Examples
    --------
    >>> criterion = ReconstructionLoss()
    >>> loss = criterion(X, S, A)
    """
    def __init__(self):
        super(ReconstructionLossNMF, self).__init__()
        self.criterion = Fro_Norm()
    def forward(self, X, S, A):
        """
        Runs the forward pass of the ReconstructionLoss module
        Parameters
        ----------
        X: Pytorch tensor
            The first input to the loss function (m x n matrix).
        A: Pytorch tensor
            The first factor of the second input to the loss function (m x k matrix).
        S: Pytorch tensor
            The second factor of the second input to the loss function (k x n matrix).
        Returns
        -------
        reconstructionloss: Pytorch tensor
            The loss of X and A*S, stored in a 1x1 Pytorch tensor to preserve information for 
            backpropagation.
        """
        X_approx = torch.mm(A,S)
        reconstructionloss = self.criterion(X_approx, X)
        return reconstructionloss


# In[ ]:


class ClassificationLossL2(nn.Module):
    '''
    calculate the classification loss, using the criterion ||L.*(Y - Y_pred)||_F^2.
    Do: criterion = ReconstructionLoss()
        loss = criterion(Y, Y_pred) and the loss is the entrywise average of the square of Frobenius norm ||Y - Y_pred||_F^2.
        loss = criterion(Y, Y_pred, L) and the loss is the entrywise average of the square of the Frobenius norm ||L.*(Y - Y_pred)||_F^2
    '''
    def __init__(self):
        super(ClassificationLossL2, self).__init__()
        self.criterion = Fro_Norm()
    def forward(self, Y, Y_pred, L = None):
        if L is None:
            classificationloss = self.criterion(Y_pred, Y)
            return classificationloss
        else:
            classificationloss = self.criterion(L*Y_pred, L*Y)
            return classificationloss

class ClassificationLossCrossEntropy(nn.Module):
    '''
    calculate the classification loss, using the criterion ||L.*(Y - Y_pred)||_F^2.
    Do: criterion = ReconstructionLoss()
        loss = criterion(Y, Y_pred) and the loss is the entrywise average of the square of Frobenius norm ||Y - Y_pred||_F^2.
        loss = criterion(Y, Y_pred, L) and the loss is the entrywise average of the square of the Frobenius norm ||L.*(Y - Y_pred)||_F^2
    '''
    def __init__(self):
        super(ClassificationLossCrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, Y_pred, label, L = None):
        if L is None:
            classificationloss = self.criterion(Y_pred, label)
            return classificationloss
        else:
            l = Variable(L[:,0].data.long())
            classificationloss = self.criterion(L*Y_pred, l*label)
            return classificationloss


# In[ ]:


class L21_Norm(nn.Module):
    '''
    Defining the L21 Norm: ||X||_{2,1} = \sum ||X_i||_2
    This norm is defined to encourage row sparsity
    '''
    def __init__(self):
        super(L21_Norm, self).__init__()
        self.criterion = nn.MSELoss()
    def forward(self, S):
        total = 0
        n = S.shape[1]
        for i in range(n):
            total += torch.norm(S[:,i])
        return total

