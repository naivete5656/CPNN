import warnings
from scvi import settings

import jax.numpy as jnp

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Gamma, constraints
from torch.distributions import Poisson as PoissonTorch

from torch.distributions.constraints import Constraint
from torch.distributions.utils import (
    broadcast_all,
    probs_to_logits,
)


class _Optional(Constraint):
    def __init__(self, constraint: Constraint):
        self.constraint = constraint

    def check(self, value: torch.Tensor) -> torch.Tensor:
        if value is None:
            return torch.ones(1, dtype=torch.bool)
        return self.constraint.check(value)

    def __repr__(self) -> str:
        return f"Optional({self.constraint})"


def optional_constraint(constraint: Constraint) -> Constraint:
    """Returns a wrapped constraint that allows optional values."""
    return _Optional(constraint)


def _convert_counts_logits_to_mean_disp(
    total_count: torch.Tensor, logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """NB parameterizations conversion.

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    logits
        success logits.

    Returns
    -------
    type
        the mean and inverse overdispersion of the NB distribution.

    """
    theta = total_count
    mu = logits.exp() * theta
    return mu, theta


def torch_lgamma_mps(x: torch.Tensor) -> torch.Tensor:
    """Used in mac Mx devices while broadcasting a tensor

    Parameters
    ----------
    x
        Data

    Returns
    -------
    lgamma tensor that perform on a copied version of the tensor
    """
    return torch.lgamma(x.contiguous())


def log_zinb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
) -> torch.Tensor:
    """Log likelihood (scalar) of a minibatch according to a zinb model.

    Parameters
    ----------
    x
        Data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    pi
        logit of the dropout parameter (real support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function

    Notes
    -----
    We parametrize the bernoulli using the logits, hence the softplus functions appearing.
    """
    log = log_fn
    lgamma = lgamma_fn
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless
    # of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    # Uses log(sigmoid(x)) = -softplus(-x)
    softplus_pi = F.softplus(-pi)
    log_theta_eps = log(theta + eps)
    log_theta_mu_eps = log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return res


def log_nb_positive(
    x: torch.Tensor | jnp.ndarray,
    mu: torch.Tensor | jnp.ndarray,
    theta: torch.Tensor | jnp.ndarray,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
) -> torch.Tensor | jnp.ndarray:
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res


def _gamma(theta: torch.Tensor, mu: torch.Tensor, on_mps: bool = False) -> Gamma:
    concentration = theta
    rate = theta / mu
    # Important remark: Gamma is parametrized by the rate = 1/scale!
    gamma_d = (
        Gamma(concentration=concentration.to("cpu"), rate=rate.to("cpu"))
        if on_mps  # TODO: NEED TORCH MPS FIX for 'aten::_standard_gamma'
        else Gamma(concentration=concentration, rate=rate)
    )
    return gamma_d


class NegativeBinomial(Distribution):
    r"""Negative binomial distribution.

    One of the following parameterizations must be provided:

    (1), (`total_count`, `probs`) where `total_count` is the number of failures until
    the experiment is stopped and `probs` the success probability. (2), (`mu`, `theta`)
    parameterization, which is the one used by scvi-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.

    In the (`mu`, `theta`) parameterization, samples from the negative binomial are generated as
    follows:

    1. :math:`w \sim \textrm{Gamma}(\underbrace{\theta}_{\text{shape}},
       \underbrace{\theta/\mu}_{\text{rate}})`
    2. :math:`x \sim \textrm{Poisson}(w)`

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    probs
        The success probability.
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    scale
        Normalized mean expression of the distribution.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": optional_constraint(constraints.greater_than_eq(0)),
        "theta": optional_constraint(constraints.greater_than_eq(0)),
        "scale": optional_constraint(constraints.greater_than_eq(0)),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
        theta: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        validate_args: bool = False,
    ):
        self.on_mps = (
            mu.device.type == "mps"
            if total_count is None
            else total_count.device.type == "mps"
        )  # TODO: This is used until torch will solve the MPS issues
        self._eps = 1e-8
        if (mu is None) == (total_count is None):
            raise ValueError(
                "Please use one of the two possible parameterizations. Refer to the documentation "
                "for more information."
            )

        using_param_1 = total_count is not None and (
            logits is not None or probs is not None
        )
        if using_param_1:
            logits = logits if logits is not None else probs_to_logits(probs)
            total_count = total_count.type_as(logits)
            total_count, logits = broadcast_all(total_count, logits)
            if self.on_mps:  # TODO: This is used until torch will solve the MPS issues
                total_count, logits = total_count.contiguous(), logits.contiguous()
            mu, theta = _convert_counts_logits_to_mean_disp(total_count, logits)
            scale = mu / torch.sum(mu, dim=-1, keepdim=True)
        else:
            mu, theta = broadcast_all(mu, theta)
            if self.on_mps:
                mu, theta = mu.contiguous(), theta.contiguous()
        self.mu = mu
        self.theta = theta
        self.scale = scale
        super().__init__(validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    def get_normalized(self, key) -> torch.Tensor:
        if key == "mu":
            return self.mu
        elif key == "scale":
            return self.scale
        else:
            raise ValueError(f"normalized key {key} not recognized")

    @property
    def variance(self) -> torch.Tensor:
        return self.mean + (self.mean**2) / self.theta

    @torch.inference_mode()
    def sample(
        self,
        sample_shape: torch.Size | tuple | None = None,
    ) -> torch.Tensor:
        """Sample from the distribution."""
        sample_shape = sample_shape or torch.Size()
        gamma_d = self._gamma()  # TODO: TORCH MPS FIX - DONE ON CPU CURRENTLY
        p_means = gamma_d.sample(sample_shape)

        # Clamping as distributions objects can have buggy behaviors when
        # their parameters are too high
        l_train = torch.clamp(p_means, max=1e8)
        counts = (
            PoissonTorch(l_train).sample().to("mps")
            if self.on_mps  # TODO: NEED TORCH MPS FIX for 'aten::poisson'
            else PoissonTorch(l_train).sample()
        )  # Shape : (n_samples, n_cells_batch, n_vars)
        return counts

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )

        lgamma_fn = (
            torch_lgamma_mps if self.on_mps else torch.lgamma
        )  # TODO: TORCH MPS FIX
        return log_nb_positive(
            value, mu=self.mu, theta=self.theta, eps=self._eps, lgamma_fn=lgamma_fn
        )

    def _gamma(self) -> Gamma:
        return _gamma(self.theta, self.mu, self.on_mps)

    def __repr__(self) -> str:
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        args_string = ", ".join(
            [
                f"{p}: "
                f"{self.__dict__[p] if self.__dict__[p].numel() == 1 else self.__dict__[p].size()}"
                for p in param_names
                if self.__dict__[p] is not None
            ]
        )
        return self.__class__.__name__ + "(" + args_string + ")"


def pearson_corr_2d(x, y, eps=1e-16):
    """
    x, y: 2D tensors (n_samples, n_features) with the same shape.
    Computes Pearson correlation for each column and returns a tensor of shape (n_features,).
    """
    # Center each feature by subtracting its mean
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)

    # Numerator: sum of covariance for each feature
    numerator = (x_centered * y_centered).sum(dim=0)
    # Denominator: product of standard deviations for each feature
    denominator = torch.sqrt((x_centered**2).sum(dim=0) * (y_centered**2).sum(dim=0))
    return numerator / (denominator + eps)
