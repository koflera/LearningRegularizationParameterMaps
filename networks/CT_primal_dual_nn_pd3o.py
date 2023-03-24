import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader  
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from odl.contrib.torch import OperatorModule
import dival
from dival import get_standard_dataset

import os
import sys
sys.path.append('../../')
sys.path.append('.')

from .grad_ops import GradOperators
from .prox_ops import ClipAct
from encoding_objects.ct_enc_object_lodopab import CTEncObj2D

from utils.ct_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
class CT_PD3O_NN(nn.Module):
    """
    Defines the class for training and evaluating the CNN which is unrolled PD3O
    """
    def __init__(self, EncObj, nu, CNN_block, mode, net_scaling = 8e+3):
        super().__init__()
        #scaling factor for the network output
        self.net_scaling = torch.tensor(net_scaling)

        #CT encoding objects
        self.EncObj  = EncObj
        self.N0 = 4096
        self.mu = 81.35858
        self.mode = mode
        
        #loss for whole network
        self.loss = nn.MSELoss()
        
        #gradient operators and clipping function
        self.dim = 2
        self.GradOps = GradOperators(self.dim)
        self.GradOps.kernel = self.GradOps.kernel.to(device)
        
        #compute Lipschitz constant of \nabla f
        self.L = self.mu**2 * self.N0 * self.EncObj.op_norm_A**2
        self.beta = 1/self.L
        
        #function for projecting 
        self.ClipAct = ClipAct()
        
        #number of iterations
        self.nu = nu
                
        #(log) constants depending on the operators
        self.tau = nn.Parameter(torch.tensor(10.,device=device),requires_grad=True) #starting value approximately  1/L
        self.sigma = nn.Parameter(torch.tensor(10.,device=device),requires_grad=True) #starting value approximately  1/L

        if mode == 'lambda_xy': 
            #one single lambda for x,y
            lambda_log_init = torch.log(torch.tensor(1000,device=device))
            self.lambda_reg_log = nn.Parameter(lambda_log_init,requires_grad=True)
        elif mode == 'lambda_cnn':
            #the CNN-block to estimate the lambda regularization map
            self.cnn = CNN_block

    def get_lambda_cnn(self, x_in):
        """
        Generate the lambda regularization map for given initial reconstruction
        """
        #compute the regularizing map 
        npad_xy = 4
        pad = (npad_xy, npad_xy, npad_xy, npad_xy)
        x_in = F.pad(x_in, pad ,mode='reflect')
        
        #estimate parameter map
        lambda_cnn = self.cnn(x_in)
        #crop
        lambda_cnn = lambda_cnn[:,:,npad_xy:-npad_xy, npad_xy:-npad_xy]
        
        #constrain map to be strictly positive
        lambda_reg = self.net_scaling * self.EncObj.op_norm_ATA * torch.nn.functional.softplus(lambda_cnn)
        return lambda_reg
    
    def forward(self, y):
        """
        Apply a forward step of the unrolled CNN
        """
        #initial reconstruction
        y = y.to(device)
        xbar = self.EncObj.apply_fbp(y).to(device)
        mb,_,Ny,Nx = xbar.shape
        
        #dual variable
        p = xbar.clone()
        q = torch.zeros(mb,self.dim,Ny,Nx, dtype = xbar.dtype).to(device)
        
        #sigma, tau
        tau = torch.tensor((2*self.beta),device=device) * torch.sigmoid(self.tau)
        sigma = (1 / tau / 8 ) * torch.sigmoid(self.sigma)   #\in (0,1/L)

        #distinguish between the different cases
        if self.mode == 'lambda_xy':
            lambda_reg = torch.exp(self.lambda_reg_log)     #\in (0,\infty)
        elif self.mode == 'lambda_cnn':
            lambda_reg = self.get_lambda_cnn(xbar)

        #compute the adjoints only once
        adj_y = self.EncObj.apply_adjoint(self.mu * self.N0 * torch.exp(-y * self.mu))
        adj_x = self.EncObj.apply_adjoint(-self.mu * self.N0 * torch.exp(-self.EncObj.apply_forward(p) * self.mu))
        for ku in range(self.nu):
            q_bar = q + sigma * self.GradOps.apply_G(xbar)
            q = self.ClipAct(q_bar,lambda_reg)
            
            p_tilde = p - tau * (adj_x + adj_y) - tau * self.GradOps.apply_GH(q)
            
            p_new = nn.ReLU()(p_tilde)
            
            adj_x_new = self.EncObj.apply_adjoint(-self.mu * self.N0 * torch.exp(-self.EncObj.apply_forward(p_new) * self.mu))
            xbar = 2 * p_new - p + tau * (adj_x - adj_x_new)
            p = p_new  
            adj_x = adj_x_new    
        return xbar
    
    def train(self,epochs,batch_size,train_size,learning_rate):
        """
        Train the CNN
        """
        #create dataset
        dataset = get_standard_dataset('lodopab', impl='astra_cuda')
        train = dataset.create_torch_dataset(part='train',
                            reshape=((1,) + dataset.space[0].shape,
                            (1,) + dataset.space[1].shape))
        val = dataset.create_torch_dataset(part='validation',
                            reshape=((1,) + dataset.space[0].shape,
                            (1,) + dataset.space[1].shape))

        val_size = 10
        train_set = []
        for i in range(train_size):
            train_set.append([train[i][0],train[i][1]])
        train = train_set
        
        val_set = []
        for i in range(val_size):
            val_set.append([val[i][0],val[i][1]])
        val = val_set
        
        dl = DataLoader(train, batch_size=batch_size,num_workers=0,pin_memory=True, shuffle=True) 
        val = next(iter(DataLoader(val_set, batch_size=val_size,num_workers=0,pin_memory=True)))
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        val_list = [] #for validation loss
        MSE_list = [] #for training loss
        val_min = 1e+10
        
        draw_new = True
        rec = recompute_observations()
        for t in tqdm(range(epochs)):
            MSE = []
            #generate new observations
            if draw_new:
                train_set = []
                for i in range(train_size):
                    obs = rec(train[i][1]).squeeze(1)
                    train_set.append([obs,train[i][1]])
                train = train_set
                dl = DataLoader(train, batch_size=batch_size,num_workers=0,pin_memory=True,shuffle=True) 
                
            for y,x in tqdm(iter(dl)):
                #compute forward mode of CNN
                pred = self.forward(y)
                
                loss = self.loss(pred,x.to(device))
                
                #backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                MSE.append(loss.item())
                
            MSE_list.append(np.array(MSE).mean())
            
            step = 2
            if (t)%step == 0:
                if not os.path.isdir('checkpoints'):
                    os.mkdir('checkpoints')   
                print(f'Validation step {t+1} \n-----------------------')
                with torch.no_grad():
                    pred = self.forward(val[0])
                    val_loss = self.loss(pred,val[1].to(device))
                    #save weights of networks for best validation config
                    print(val_loss)
                    if val_loss < val_min:
                        print(val_min)
                        torch.save({'net_state_dict': self.state_dict(), 'iter': self.nu}, f'checkpoints/pd3o_weights_net.pth')
                        val_min = val_loss
                    val_list.append(val_loss.item())
                    
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.plot(list(range(len(MSE_list))), MSE_list, 'k-', label='MSE train')
                    plt.plot(list(range(0,len(val_list)*step,step)), val_list, 'k-.', label='MSE validation')
                    plt.legend(loc='upper right')
                    plt.yscale('log')
                    plt.savefig(f'checkpoints/losscurve_{self.mode}_pd3o.pdf')
                    plt.close()
                print(f'-------------------------------')
                
