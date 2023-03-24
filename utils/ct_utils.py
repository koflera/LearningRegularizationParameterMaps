import torch
from torch import nn
import odl
import numpy as np

from skimage.transform import resize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class recompute_observations(nn.Module):
    '''
    class to compute the noisy sinogram for a given CT scan,
    see file:///homes/numerik/altekrue/Downloads/s41597-021-00893-z.pdf
    for the corresponding description
    '''
    def __init__(self):
        super().__init__()
        self.photons = 4096
        self.angles = 1000
        self.num_det_pixels = 513
        self.im_shape = (1000,1000)
        self.MIN_PT = [-0.13, -0.13]
        self.MAX_PT = [0.13, 0.13]
        self.mu = 81.35858
        self.impl = 'astra_cuda'
        self.rs = np.random.RandomState()

        self.space = odl.uniform_discr(min_pt=self.MIN_PT, max_pt=self.MAX_PT, shape=self.im_shape,
                                  dtype=np.float64)

        self.geometry = odl.tomo.parallel_beam_geometry(
            self.space, num_angles=self.angles, det_shape=(self.num_det_pixels,))

        self.ray_trafo = odl.tomo.RayTransform(self.space, self.geometry, impl=self.impl)
        
    def forward(self,tens):
        im = tens.squeeze().cpu().numpy()
        im_resized = resize(im * self.mu, self.im_shape, order=1)
        data = self.ray_trafo(im_resized).asarray()
        data *= (-1)
        np.exp(data, out=data)
        data *= self.photons
            
        data = self.rs.poisson(data) / self.photons
        np.maximum(0.1 / self.photons, data, out=data)
        np.log(data, out=data)
        data /= (-self.mu)
        return torch.tensor(data).reshape(1,1,1000,513)
