import torch


def z_score(
    x: torch.Tensor, dim: int, keepdim: bool = True, unbiased: bool = True
) -> torch.Tensor:
    x_mean = x.mean(dim=dim, keepdim=keepdim)
    x_std = x.std(dim=dim, keepdim=keepdim, unbiased=unbiased)

    x = (x - x_mean) / x_std
    return x
