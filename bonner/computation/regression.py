from typing import Union

import torch
import torch.nn as nn


class LinearRegression:
    def __init__(
        self, fit_intercept: bool = True, device: Union[torch.device, str] = None
    ) -> None:
        self.fit_intercept = fit_intercept
        self.device_ = device
        self.n_features_in_ = None
        self.coef_ = None
        self.intercept_ = None
        self._residues = None
        self.rank_ = None
        self.singular_ = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.to(self.device_)
        n_samples_, self.n_features_in_ = x.shape

        y = y.to(self.device_)
        if self.fit_intercept:
            x = torch.cat(
                [x, torch.ones(n_samples_, device=self.device_).unsqueeze(1)], dim=1
            )

        self.coef_, self._residues, self.rank_, self.singular_ = torch.linalg.lstsq(
            x, y
        )

        if self.fit_intercept:
            self.intercept_ = self.coef_[-1, :]
        else:
            self.intercept_ = torch.zeros(self.n_features_in_)

        self.coef_ = self.coef_[:-1, :].transpose_(0, 1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.coef_ is None:
            raise RuntimeError("model has not been fit")
        else:
            x = x.to(self.device_)
            x = torch.matmul(x, self.coef_.transpose_(0, 1))
            x += self.intercept_
            return x


class RidgeRegression:
    # adapted from https://gist.github.com/myazdani/3d8a00cf7c9793e9fead1c89c1398f12
    def __init__(
        self,
        regularization: float = 1,
        fit_intercept: bool = True,
        device: Union[torch.device, str] = None,
    ) -> None:
        self.regularization = regularization
        self.fit_intercept = fit_intercept
        self.device_ = device

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.to(self.device_)
        y = y.to(self.device_)

        assert x.shape[0] == y.shape[0], "number of X and y rows don't match"
        if self.fit_intercept:
            x = torch.cat([torch.ones(x.shape[0], 1, device=self.device_), x], dim=1)

        lhs = x.T @ x  # FIXME I don't think this is correct
        rhs = x.T @ y
        if self.regularization == 0:
            self.w, _ = torch.lstsq(rhs, lhs)
        else:
            ridge = self.regularization * torch.eye(lhs.shape[0], device=self.device_)
            self.w, _ = torch.lstsq(rhs, lhs + ridge)

    def predict(self, x: torch.tensor) -> torch.Tensor:
        x = x.to(self.device_)
        if self.fit_intercept:
            x = torch.cat([torch.ones(x.shape[0], 1, device=self.device_), x], dim=1)
        return x @ self.w


class LogisticRegression(nn.Module):
    def __init__(self, n_classes: int, device: torch.device = None) -> None:
        super().__init__()
        if device is not None:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.n_classes = n_classes

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer_kwargs={
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        },
        n_epochs: int = 100,
    ) -> None:
        self.model = nn.Linear(
            x.shape[1], self.n_classes, bias=True, device=self.device
        )
        x = x.float().to(self.device)
        y = x.long().to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_kwargs)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(n_epochs):
            predictions = self.model(x)
            loss = loss_fn(predictions, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, x: torch.Tensor, output="labels") -> torch.Tensor:
        x = x.float().to(self.device)
        logits = self.model(x)
        if output == "logits":
            return logits.cpu()
        scores = torch.softmax(logits, dim=1)
        if output == "scores":
            return scores.detach().cpu()
        labels = torch.argmax(scores, dim=1)
        if output == "labels":
            return labels.detach().cpu()
