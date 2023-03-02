import torch
from .linalg import inner_product


def conj_grad(H, x, b, niter=4, tol=1e-16):
    r = H(x)
    r = b - r
    p = r.clone()
    sqnorm_r_old = inner_product(r, r)

    for kiter in range(niter):
        d = H(p)
        inner_p_d = inner_product(p, d)
        alpha = sqnorm_r_old / inner_p_d
        x = torch.add(x, p, alpha=alpha.item())
        r = torch.add(r, d, alpha=-alpha.item())
        sqnorm_r_new = inner_product(r, r)
        beta = sqnorm_r_new / sqnorm_r_old
        sqnorm_r_old = sqnorm_r_new
        p = torch.add(r, p, alpha=beta.item())
    return x
