import numpy as np
import torch


class _BasePCA:
    """Based on sklearn's _BasePCA class.

    https://github.com/scikit-learn/scikit-learn/blob/37ac6788c9504ee409b75e5e24ff7d86c90c2ffb/sklearn/decomposition/_base.py#L19
    """

    def __init__(
        self,
        n_components: int = None,
        *,
        device: torch.device | str | None = None,
        whiten: bool = False
    ):
        self.n_components_ = n_components
        self.whiten = whiten
        self.device = device

        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.noise_variance_ = None
        self.mean_ = None
        self.var_ = None
        self.n_samples_ = None
        self.n_features_ = None

    def get_covariance(self) -> torch.Tensor:
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * torch.sqrt(exp_var).unsqueeze(-1)
        exp_var_diff = torch.maximum(exp_var - self.noise_variance_, torch.tensor(0))
        cov = torch.matmul(components_.transpose(0, 1) * exp_var_diff, components_)
        cov.flatten()[:: len(cov) + 1] += self.noise_variance_  # modify diag inplace
        return cov

    def get_precision(self) -> torch.Tensor:
        n_features = self.components_.shape[1]

        # get precision using matrix inversion lemma
        if self.n_components_ == 0:
            return torch.eye(n_features) / self.noise_variance_
        if self.n_components_ == n_features:
            return torch.linalg.inv(self.get_covariance())

        # Get precision using matrix inversion lemma
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * torch.sqrt(exp_var.unsqueeze(-1))
        exp_var_diff = torch.maximum(exp_var - self.noise_variance_, torch.tensor(0))
        precision = (
            torch.matmul(components_, components_.transpose(0, 1))
            / self.noise_variance_
        )
        precision.flatten()[:: len(precision) + 1] += 1 / exp_var_diff
        precision = torch.matmul(
            components_.transpose(0, 1),
            torch.matmul(torch.linalg.inv(precision), components_),
        )
        precision /= -(self.noise_variance_**2)
        precision.flatten()[:: len(precision) + 1] += 1 / self.noise_variance_
        return precision

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if self.mean_ is not None:
            x = x - self.mean_
        x_transformed = torch.matmul(x, self.components_.transpose(0, 1))
        if self.whiten:
            x_transformed /= torch.sqrt(self.explained_variance_)
        return x_transformed

    def inverse_transform(
        self, x: torch.Tensor, n_components: int = None
    ) -> torch.Tensor:
        if n_components is None:
            if self.whiten:
                return (
                    torch.matmul(
                        x,
                        torch.sqrt(self.explained_variance_.unsqueeze(-1))
                        * self.components_,
                    )
                    + self.mean_
                )
            else:
                return torch.matmul(x, self.components_) + self.mean_
        else:
            if self.whiten:
                return (
                    torch.matmul(
                        x[:, :n_components],
                        torch.sqrt(
                            self.explained_variance_[:n_components].unsqueeze(-1)
                        )
                        * self.components_[:n_components, :],
                    )
                    + self.mean_
                )
            else:
                return (
                    torch.matmul(
                        x[:, :n_components], self.components_[:n_components, :]
                    )
                    + self.mean_
                )


class PCA(_BasePCA):
    """Based on sklearn's PCA class.

    https://github.com/scikit-learn/scikit-learn/blob/37ac6788c9504ee409b75e5e24ff7d86c90c2ffb/sklearn/decomposition/_pca.py#L116
    """

    def fit(self, x: torch.Tensor) -> None:
        self.n_samples_, self.n_features_ = x.shape
        if self.n_components_ is None:
            self.n_components_ = min(self.n_samples_, self.n_features_)

        x = x.to(self.device)
        self.mean_ = torch.mean(x, dim=0)

        x -= self.mean_

        u, s, v_h = torch.linalg.svd(x, full_matrices=False)
        u, v_h = _svd_flip(u, v_h)
        explained_variance = (s**2) / (self.n_samples_ - 1)

        self.components_ = v_h[: self.n_components_]
        self.explained_variance_ = explained_variance[: self.n_components_]
        self.explained_variance_ratio_ = (
            explained_variance / explained_variance.sum()
        )[: self.n_components_]
        self.singular_values_ = s[: self.n_components_]
        if self.n_components_ < x.shape[1]:
            self.noise_variance_ = explained_variance[self.n_components_ :].mean()
        else:
            self.noise_variance_ = 0


class IncrementalPCA(_BasePCA):
    """Based on sklearn's IncrementalPCA class:

    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
    """

    def __init__(self, *args, **kwargs) -> None:
        super(IncrementalPCA, self).__init__(*args, **kwargs)

        self._initialized: bool = False
        self._batch_size = None
        self.n_samples_seen_ = None

    def fit_partial(self, x: torch.Tensor) -> None:
        x = x.to(self.device)
        if not self._initialized:
            self._initialize_from(x)
        n_feats = x.size(1)

        new_mean, new_var, new_n_samples_seen = self._incremental_mean_and_var(x)

        # Whitening
        if self.n_samples_seen_ == 0:
            # If this is the first step, simply whitten X
            x -= new_mean
        else:
            batch_mean = x.mean(dim=0)
            x -= batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = np.sqrt(
                (self.n_samples_seen_ / new_n_samples_seen) * x.size(0)
            ) * (self.mean_ - batch_mean)
            x = torch.vstack(
                [
                    self.singular_values_.reshape(-1, 1) * self.components_,
                    x,
                    mean_correction,
                ]
            )

        u, s, v_t = torch.linalg.svd(x, full_matrices=False)
        u, v_t = _svd_flip(u, v_t, u_based_decision=False)
        s_squared = s**2
        explained_variance = s_squared / (new_n_samples_seen - 1)
        explained_variance_ratio = s_squared / torch.sum(new_var * new_n_samples_seen)

        self.mean_ = new_mean
        self.var_ = new_var
        self.n_samples_seen_ = new_n_samples_seen

        self.components_ = v_t[: self.n_components_]
        self.singular_values_ = s[: self.n_components_]
        self.explained_variance_ = explained_variance[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_]
        if self.n_components_ < n_feats:
            self.noise_variance_ = explained_variance[self.n_components_ :].mean()
        else:
            self.noise_variance_ = 0

    def _initialize_from(self, X: torch.Tensor) -> None:
        assert X.ndim == 2

        batch_size, n_feats = X.shape
        if self.n_components_ is None:
            self.n_components_ = min(batch_size, n_feats)
        else:
            assert 1 <= self.n_components_ <= min(batch_size, n_feats)
        self._batch_size, self.n_features_ = batch_size, n_feats

        self.mean_ = torch.zeros(n_feats).to(self.device)
        self.var_ = torch.zeros(n_feats).to(self.device)
        self.n_samples_seen_ = 0

        self.singular_values_ = torch.zeros(self.n_components_).to(self.device)
        self.components_ = torch.zeros(self.n_components_, n_feats).to(self.device)

        self._initialized = True

    def _incremental_mean_and_var(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        X_sum = X.sum(dim=0)
        X_n_samples = X.size(0)
        last_sum = self.mean_ * self.n_samples_seen_

        new_n_samples_seen = self.n_samples_seen_ + X_n_samples
        new_mean = (last_sum + X_sum) / new_n_samples_seen

        temp = X - X_sum / X_n_samples
        correction = temp.sum(dim=0)
        X_unnormalized_var = (temp**2).sum(dim=0)
        X_unnormalized_var -= correction**2 / X_n_samples
        last_unnormalized_var = self.var_ * self.n_samples_seen_
        last_over_new_n_samples = max(self.n_samples_seen_ / X_n_samples, 1e-7)
        new_unnormalized_var = (
            last_unnormalized_var
            + X_unnormalized_var
            + last_over_new_n_samples
            / new_n_samples_seen
            * (last_sum / last_over_new_n_samples - X_sum) ** 2
        )
        new_var = new_unnormalized_var / new_n_samples_seen

        return new_mean, new_var, new_n_samples_seen


def _svd_flip(
    u: torch.Tensor, v: torch.Tensor, u_based_decision: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    if u_based_decision:
        max_abs_cols = torch.argmax(torch.abs(u), dim=1)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs.unsqueeze(dim=1)
    else:
        max_abs_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs.unsqueeze(dim=1)
    return u, v
