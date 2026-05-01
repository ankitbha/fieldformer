from __future__ import annotations

import torch

try:
    import gpytorch
    from gpytorch.kernels import PeriodicKernel, ProductKernel, RBFKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
    from gpytorch.means import ConstantMean, ZeroMean
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import (
        CholeskyVariationalDistribution,
        IndependentMultitaskVariationalStrategy,
        VariationalStrategy,
    )
except Exception:  # pragma: no cover - import-time convenience for py_compile-only envs
    gpytorch = None
    ApproximateGP = object
    GaussianLikelihood = None
    MultitaskGaussianLikelihood = None


class PeriodicSVGP(ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        if gpytorch is None:
            raise ImportError("gpytorch is required for SVGP baselines")
        m = inducing_points.size(0)
        q = CholeskyVariationalDistribution(m)
        vs = VariationalStrategy(self, inducing_points, q, learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module = ConstantMean()
        kx, ky, kt = PeriodicKernel(), PeriodicKernel(), PeriodicKernel()
        kx.initialize(period_length=1.0)
        ky.initialize(period_length=1.0)
        kt.initialize(period_length=1.0)
        self.covar_module = ScaleKernel(ProductKernel(kx, ky, kt))

    def forward(self, x: torch.Tensor):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class MultitaskPeriodicSVGP(ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, num_tasks: int = 3):
        if gpytorch is None:
            raise ImportError("gpytorch is required for SVGP baselines")
        batch_shape = torch.Size([num_tasks])
        inducing = inducing_points.unsqueeze(0).expand(num_tasks, -1, -1).contiguous()
        q = CholeskyVariationalDistribution(inducing.size(-2), batch_shape=batch_shape)
        base_vs = VariationalStrategy(self, inducing, q, learn_inducing_locations=True)
        vs = IndependentMultitaskVariationalStrategy(base_vs, num_tasks=num_tasks)
        super().__init__(vs)
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        kx = PeriodicKernel(batch_shape=batch_shape)
        ky = PeriodicKernel(batch_shape=batch_shape)
        kt = PeriodicKernel(batch_shape=batch_shape)
        kx.initialize(period_length=1.0)
        ky.initialize(period_length=1.0)
        kt.initialize(period_length=1.0)
        self.covar_module = ScaleKernel(ProductKernel(kx, ky, kt), batch_shape=batch_shape)

    def forward(self, x: torch.Tensor):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class PollutionSVGP(ApproximateGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        ard_lengthscale_init: tuple[float, float, float] = (0.2, 0.2, 0.1),
        outputscale_init: float = 1.0,
    ):
        if gpytorch is None:
            raise ImportError("gpytorch is required for SVGP baselines")
        m, d = inducing_points.shape
        q = CholeskyVariationalDistribution(m)
        vs = VariationalStrategy(self, inducing_points, q, learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=d))
        with torch.no_grad():
            self.covar_module.base_kernel.lengthscale = torch.tensor(
                ard_lengthscale_init, dtype=torch.float32, device=inducing_points.device
            )
            self.covar_module.outputscale = torch.tensor(
                float(outputscale_init), dtype=torch.float32, device=inducing_points.device
            )

    def forward(self, x: torch.Tensor):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def make_likelihood(dataset_key: str):
    if gpytorch is None:
        raise ImportError("gpytorch is required for SVGP baselines")
    if dataset_key == "swe":
        return MultitaskGaussianLikelihood(num_tasks=3)
    return GaussianLikelihood()

