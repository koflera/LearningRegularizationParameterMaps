import torch
from .linalg import inner_product

def conj_grad(H, x, b, niter=4, tol=1e-16):
		
	#x is the starting value, b the rhs;
	r = H(x)
	r = b-r
	
	#initialize p
	p = r.clone()
	
	#old squared norm of residual
	sqnorm_r_old = inner_product(r,r)
	
	#sqnorm_b = inner_product(b,b)
	
	for kiter in range(niter):
		
		#print(kiter)
		#if sqnorm_r_old.item() / sqnorm_b.item()  > tol:
		
		#calculate Hp;
		d = H(p)

		#calculate step size alpha;
		inner_p_d = inner_product(p, d)
		
		alpha = sqnorm_r_old / inner_p_d

		#perform step and calculate new residual;
		x = torch.add(x, p, alpha= alpha.item())
		r = torch.add(r, d, alpha= -alpha.item())
		
		#new residual norm
		sqnorm_r_new = inner_product(r,r)
		#print('k={}; ||r||_2 = {}'.format(kiter, sqnorm_r_new))
		
		#calculate beta and update the norm;
		beta = sqnorm_r_new / sqnorm_r_old
		sqnorm_r_old = sqnorm_r_new

		p = torch.add(r,p,alpha=beta.item())
		#kiter+=1
		
	return x