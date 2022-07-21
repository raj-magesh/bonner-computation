from typing import Callable
import pickle
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


EPSILON = torch.finfo(torch.float32).eps


class AsgMuNmf(Dataset):
    """
    AsgMuNmf performs asymmetric gradient multiplicative update nonnegative-matrix factorization
    """

    def __init__(self, data: np.ndarray, n_components: int) -> None:
        self.data = data
        self.n_samples = data.shape[0]
        self.n_dimensions = data.shape[1]
        self.n_components = n_components

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return idx

    def collate_samples(self, indices):
        indices = np.array(indices, dtype=np.int64)
        rows = torch.from_numpy(self.data[indices, :].toarray())
        indices = torch.from_numpy(indices).long()
        return rows, indices

    def fit_transform(
        self,
        u: torch.Tensor = None,
        v: torch.Tensor = None,
        n_epochs: int = 200,
        batch_size: int = 10000,
        num_workers: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        if u is None:
            u = torch.normal(0, 0.01, size=(self.n_samples, self.n_components)).abs()
        if v is None:
            v = torch.normal(0, 0.01, size=(self.n_components, self.n_dimensions)).abs()

        with open("nmf_initialization.pkl", "wb") as f:
            pickle.dump((u.cpu().numpy(), v.cpu().numpy()), f)

        u, v = u.to(device), v.to(device)
        _run_asg_mu_nmf(
            self, u, v, device, n_epochs, batch_size, num_workers, self.collate_samples
        )
        u, v = u.cpu().numpy(), v.cpu().numpy()

        return u, v


def _run_asg_mu_nmf(
    x: Dataset,
    u: torch.Tensor,
    v: torch.Tensor,
    device: torch.device | str,
    n_epochs: int,
    batch_size: int,
    num_workers: int,
    collate_fn: Callable,
) -> None:
    with torch.no_grad():
        for epoch in range(n_epochs):
            batch_loader = DataLoader(
                x,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                pin_memory=True,
                collate_fn=collate_fn,
            )

            for batch, data in enumerate(batch_loader):
                print(f"epoch {epoch} batch {batch}")

                x_batch, indices = data
                x_batch = x_batch.to(device)
                indices = indices.to(device)
                u_batch = u[indices, :]
                u[indices, :], v = _update_u_and_v(x_batch, u_batch, v)

            with open(f"nmf_epoch_{epoch}.pkl", "wb") as f:
                pickle.dump((u.cpu().numpy(), v.cpu().numpy()), f)

            if epoch > 0:
                pathlib.Path(f"nmf_epoch_{epoch - 1}.pkl").unlink()


def _update_u_and_v(
    x: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    u_t = u.transpose(0, 1)
    v_t = v.transpose(0, 1)
    uv = torch.mm(u, v)
    u *= torch.mm(x, v_t) / (torch.mm(uv, v_t) + EPSILON)
    v *= torch.mm(u_t, x) / (torch.mm(u_t, uv) + EPSILON)
    return u, v
