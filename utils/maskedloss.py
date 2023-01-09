import torch
from torch import Tensor
from typing import Callable, Optional


def maskedloss(x: Tensor, target: Tensor, mask: Optional[Tensor] = None, lossfunction: Callable = torch.nn.functional.mse_loss, maskweight: float = 0.0) -> Tensor:
    """_summary_
    Applies a loss function only on masked area
    Args:
        x (Tensor): Value
        target (Tensor): Target
        mask (Optional[Tensor], optional): Mask. Defaults to None.
        lossfunction (Callable, optional): Loss Function. Defaults to torch.nn.functional.mse_loss.
        maskweight (float, optional): Ifset to value>0., still consider loss outside mask but weight it

    Raises:
        ValueError:  x and target should be both either real or complex

    Returns:
        Tensor: Loss
    """
    if x.is_complex() and target.is_complex():
        x = torch.view_as_real(x)
        target = torch.view_as_real(target)
        if mask is not None:
            mask = mask.unsqueeze(-1)
    elif x.is_complex() or target.is_complex():
        raise ValueError("either both should be real or complex")

    if mask is None or maskweight == 1:
        return lossfunction(x, target, reduction="mean")
    elif maskweight > 0:
        err = lossfunction(x, target, reduction="none")
        masksum = (1 + maskweight) * mask.sum() * err.nelement() / mask.nelement()
        notmasksum = (1 + maskweight) * (~mask).sum() * err.nelement() / mask.nelement()
        loss1 = (err * mask).sum() / masksum
        loss2 = (err * (~mask)).sum() / notmasksum
        return loss1 + maskweight * loss2
    else:
        err = lossfunction(x, target, reduction="none")
        masksum = mask.sum() * err.nelement() / mask.nelement()
        return (err * mask).sum() / masksum
