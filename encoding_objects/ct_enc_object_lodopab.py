import torch
import torch.nn as nn
import odl
from odl.contrib import torch as odl_torch
import dival
from dival import get_standard_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CTEncObj2D(nn.Module):
    """
    Class which creates the forward operators as torch module:
    """
    def __init__(self):
        super().__init__()
        dataset = get_standard_dataset('lodopab', impl='astra_cuda')
        A = dataset.ray_trafo
        AT = A.adjoint
        
        #create FBP
        filter_type = "Hann"
        frequency_scaling = 0.641025641025641
        fbp = odl.tomo.fbp_op(A, filter_type=filter_type, frequency_scaling=frequency_scaling)
        
        #calculate relevant operator norms:
        ATA = odl.operator.operator.OperatorComp(AT,A)
        self.op_norm_A = odl.power_method_opnorm(A, maxiter=200)
        self.op_norm_ATA = odl.power_method_opnorm(ATA, maxiter=200)
        
        # Wrap ODL operators as nn modules 
        self.A_op_layer = odl_torch.OperatorModule(A).to(device)
        self.AT_op_layer = odl_torch.OperatorModule(AT).to(device)
        self.fbp_op_layer = odl_torch.OperatorModule(fbp).to(device)
                
    def apply_forward(self, x):
        return self.A_op_layer(x)
    
    def apply_adjoint(self, y):
        return self.AT_op_layer(y)
    
    def apply_fbp(self, y):
        return self.fbp_op_layer(y)


