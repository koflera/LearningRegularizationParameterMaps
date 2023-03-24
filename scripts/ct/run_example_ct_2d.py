import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from torch.utils.data import DataLoader 
import odl
from odl.contrib.torch import OperatorModule
import dival
import dival.reconstructors.networks.unet
from dival import get_standard_dataset
import argparse

import sys
sys.path.append('../../')
sys.path.append('.')

from encoding_objects.ct_enc_object_lodopab import CTEncObj2D
from networks.CT_primal_dual_nn_pd3o import CT_PD3O_NN
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default = 'lambda_cnn',
                        help='Choose between regularization parameter map (lambda_cnn) and scalar regularization parameter (lambda_xy) (default: lambda_cnn)')
    parser.add_argument('--iterations', type = int, default = 512,
                        help='Choose the number of iterations for PD3O (default: 512)')
    parser.add_argument('--train', type = bool, default = False,
                        help='Choose whether the CNN should be trained, otherwise load pretrained weights (default: False)')                    
    args = parser.parse_args()

    #load dataset
    dataset = get_standard_dataset('lodopab', impl='astra_cuda')
    ray_trafo = dataset.ray_trafo

    #init the unet
    lambda_cnn = dival.reconstructors.networks.unet.get_unet_model(in_ch =1, out_ch = 1, use_sigmoid=False).to(device)
    #load operators for CT  
    EncObj = CTEncObj2D()

    #configurations
    iterations = args.iterations
    mode = args.mode
    train = args.train
    net = CT_PD3O_NN(EncObj, nu = iterations, CNN_block = lambda_cnn, mode = mode)

    if train:
        epochs = 50
        batch_size = 1
        train_size = 300
        learning_rate = 1e-4 if mode == 'lambda_cnn' else 1e-2
        print(f'Start training of {mode} for {epochs} epochs, learning rate {learning_rate}, {train_size} training imgs and a batch size of {batch_size}.')
        net.train(epochs,batch_size,train_size,learning_rate)    
        
    else:            
        if net.mode == 'lambda_xy':
            net.load_state_dict(torch.load('results/pd3o_weights_scalar.pth')['net_state_dict'])
        else:
            #net.load_state_dict(torch.load('checkpoints/pd3o_weights_net.pth')['net_state_dict'])
            net.load_state_dict(torch.load('results/pd3o_weights_net.pth')['net_state_dict'])

        test = dataset.create_torch_dataset(part='test',
                                    reshape=((1,1,) + dataset.space[0].shape,
                                    (1,1,) + dataset.space[1].shape))
        
        net.nu = 1024
        test_img = 19
        with torch.no_grad():
            y = test[test_img][0].to(device)
            fbp = net.EncObj.apply_fbp(y).to(device)
            if net.mode == 'lambda_cnn':
                lam = np.log(np.rot90(torch.abs(net.get_lambda_cnn(fbp)).squeeze().detach().cpu().numpy(),k=3))
                plt.imshow(lam,cmap=plt.cm.inferno,clim=[0,12])
                plt.box(False)
                ax = plt.gca()
                ax.axis('off')
                plt.savefig(f'results/LambdaMap_CT.png', bbox_inches='tight', pad_inches=0)
            pred = net(y)
            img = np.clip(np.rot90(pred.squeeze().detach().cpu().numpy(),k=3),0,1)
            io.imsave(f'results/Reco_CT.png', img)
