import torch

from .z_score import z_score


def corrcoef(
    x: torch.Tensor,
    y: torch.Tensor = None,
    return_diagonal: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """A more powerful corrcoef function that computes the Pearson correlation coefficient.

    x and y optionally take a batch dimension (either x or y, or both; in the latter case, the pairwise correlations are broadcasted along the batch dimension). If x and y are both specified, pairwise correlations between the columns of x and those of y are computed.
    # TODO implement batching along batch dimension when n_batch is too large to fit in memory
    # TODO implement batching when n_features is too large for (n_features, n_features) to fit in memory

    :param x: a tensor of shape (n_batch, n_samples, n_features) or (n_samples, n_features)
    :param y: an optional tensor of shape (n_batch, n_samples, n_features) or (n_samples, n_features), defaults to None
    :param return_diagonal: when both x and y are specified and have corresponding features (i.e. equal n_features), returns only the diagonal of the (n_features_x, n_features_y) pairwise correlation matrix, defaults to True
    :param device: torch device (cpu or cuda), defaults to None
    :return: Pearson correlation coefficients
    """
    dim_sample_x, dim_feature_x = x.ndim - 2, x.ndim - 1
    n_samples = x.shape[dim_sample_x]
    x = x.to(device)
    x = z_score(x, dim=dim_sample_x)

    if y is not None:
        dim_sample_y, dim_feature_y = y.ndim - 2, y.ndim - 1
        assert (
            x.shape[dim_sample_x] == y.shape[dim_sample_y]
        ), "x and y must have same n_samples"
        if return_diagonal:
            assert (
                x.shape[dim_feature_x] == y.shape[dim_feature_y]
            ), "x and y must have same n_features to return diagonal"
        y = y.to(device)
        y = z_score(y, dim_sample_y)
    else:
        y = x

    x = torch.matmul(x.transpose(-2, -1), y) / (n_samples - 1)
    if return_diagonal:
        x = torch.diagonal(x, dim1=-2, dim2=-1)
    return x
