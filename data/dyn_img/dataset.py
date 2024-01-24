import os

import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

from .data_utils import extract_patches_3d


class DynamicImageDenoisingDataset(Dataset):
	
	def __init__(
		self, 
		data_path: str, 
		ids: list,
		scale_factor = 0.5, 
		sigma=0.23,  
		patches_size = None,
		strides= None,
		extract_data=True,
		device: str = "cuda"
	):
		self.device = device
		self.scale_factor = scale_factor

		ids = [str(x).zfill(2) for x in ids]

		xf_list = []
  
		for k, img_id in enumerate(ids):
			print(f'loading image id {img_id}, {k}/{len(ids)}')
			sample_path = os.path.join(data_path, f"MOT17-{img_id}")
			if extract_data:
				xf = self.create_dyn_img(sample_path).unsqueeze(0) / 255
			else:
				scale_factor_str = str(self.scale_factor).replace('.','_')
				xf = np.load(os.path.join(sample_path, f"xf_scale_factor{scale_factor_str}.npy"))
				xf = torch.tensor(xf, dtype=torch.float)
				xf = xf.unsqueeze(0).unsqueeze(0) / 255
			
			if patches_size is not None:
				print(f"extracting patches of shape {patches_size}; strides {strides}")
				xfp = extract_patches_3d(xf.contiguous(), patches_size, stride=strides)
				xf_list.append(xfp)
			
		if patches_size is not None:
			# will have shape (mb, 1, Nx, Ny, Nt), where mb denotes the number of patches
			xf = torch.concat(xf_list,dim=0)
			
		else:
			xf = xf.unsqueeze(0)
		
		#create temporal TV vector to detect which patches contain the most motion
		xfp_tv = (xf[...,1:] - xf[...,:-1]).pow(2).sum(dim=[1,2,3,4]) #contains the TV for all patches
		
		#normalize to 1 to have a probability vector
		xfp_tv /= torch.sum(xfp_tv)
		
		#sort TV in descending order --> xfp_tv_ids[0] is the index of the patch with the most motion
		self.samples_weights = xfp_tv
		
		self.xf = xf
		self.len = xf.shape[0]
		
		if isinstance(sigma, float):
			self.noise_level = 'constant'
			self.sigma = sigma

		elif isinstance(sigma, (tuple, list)):
			self.noise_level = 'variable'
			self.sigma_min = sigma[0]
			self.sigma_max = sigma[1]
		
		else:
			raise ValueError("Invalid sigma value provided, must be float, tuple or list.")

	def create_dyn_img(self, sample_path: str):
		
		files_path = os.path.join(sample_path, "img1")
		files_list = os.listdir(files_path)
		xf = []

		for file in files_list:
			
			image = Image.open(os.path.join(files_path, file))
			
			#resize
			Nx_,Ny_ = np.int(np.floor(self.scale_factor * image.width )), np.int(np.floor(self.scale_factor * image.height ))
			image = image.resize( (Nx_, Ny_) )
			
			#convert to grey_scale
			image = image.convert('L')
			image_data = np.asarray(image)
			xf.append(image_data)
			
		xf = np.stack(xf, axis=-1)
		
		scale_factor_str = str(self.scale_factor).replace('.','_')
		np.save(os.path.join(sample_path, f"xf_scale_factor{scale_factor_str}.npy"), xf)
		
		return torch.tensor(xf, dtype = torch.float)
			
	def __getitem__(self, index):

		std = torch.std(self.xf[index])
		mu = torch.mean(self.xf[index])

		x_centred = (self.xf[index]  - mu) / std

		if self.noise_level == 'constant':
			sigma = self.sigma
			
		elif self.noise_level == 'variable':
			sigma = self.sigma_min + torch.rand(1) * ( self.sigma_max - self.sigma_min )

		x_centred += sigma * torch.randn(self.xf[index].shape, dtype = self.xf[index].dtype)

		xnoise = std * x_centred + mu
  
		return (
			xnoise.to(device=self.device),
   			self.xf[index].to(device=self.device)
        )
		
	def __len__(self):
		return self.len