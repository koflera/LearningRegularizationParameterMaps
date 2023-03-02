import torch

def inner_product(t1,t2):
	
	if torch.is_complex(t1):
		
		innerp = torch.sum(t1.flatten() * t2.flatten().conj())
	else:
		innerp = torch.sum(t1.flatten() * t2.flatten())
	return innerp
		