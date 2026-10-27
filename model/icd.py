"""Individual change detection (ICD) evidence branches.

The Bayesian branch adapts BARO's online Bayesian anomaly-detection idea to
short, high-dimensional neural-network windows.  Instead of fitting one large
covariance matrix, it maintains an independent conjugate predictive model for
every node feature and emits a differentiable change probability.
"""
from __future__ import annotations

import torch
from torch import nn


class _EvidenceEncoder(nn.Module):
    """Map four directional summary statistics to root-cause logits."""

    def __init__(self, num_nodes: int, node_dim: int, hidden_dim: int, num_causes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_nodes * node_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_causes),
        )

    def encode(self, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            [positive.mean(1), negative.mean(1), positive.amax(1), negative.amax(1)],
            dim=-1,
        )
        return self.encoder(pooled.flatten(1))


class RobustEvidenceBranch(_EvidenceEncoder):
    """Original median/MAD ICD, retained as a reproducible comparison."""

    def __init__(self, num_nodes, node_dim, hidden_dim, num_causes, eps=1e-6):
        super().__init__(num_nodes, node_dim, hidden_dim, num_causes)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        median = x.median(dim=1, keepdim=True).values
        mad = (x - median).abs().median(dim=1, keepdim=True).values.clamp_min(self.eps)
        z = (x - median) / (1.4826 * mad)
        return self.encode(torch.relu(z), torch.relu(-z))


class BayesianOnlineEvidenceBranch(_EvidenceEncoder):
    """Exact finite-window Bayesian online changepoint detection (BOCD).

    For every node feature this module maintains the complete run-length
    posterior ``p(r_t | x_1:t)``.  A Normal-Inverse-Gamma conjugate model gives
    a Student-t posterior predictive for every run-length hypothesis.  At each
    observation the posterior is split into changepoint and growth messages by
    a constant hazard function, normalised in log space, and its sufficient
    statistics are updated online.  This is the actual Adams-MacKay BOCD
    recursion used by BARO, implemented in PyTorch for end-to-end training.

    Since ``P(r_t=0)`` is equal to a constant hazard under the standard BOCD
    transition, ICD uses posterior mass on short run lengths (0, 1, 2) as the
    differentiable change evidence.  The sign comes from the pre-update
    posterior-predictive residual.
    """

    def __init__(
        self,
        num_nodes,
        node_dim,
        hidden_dim,
        num_causes,
        hazard=1.0 / 50.0,
        prior_kappa=0.1,
        prior_alpha=2.0,
        prior_beta=1.0,
        short_run=2,
        eps=1e-6,
    ):
        super().__init__(num_nodes, node_dim, hidden_dim, num_causes)
        self.hazard = float(hazard)
        self.prior_kappa = float(prior_kappa)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.short_run = int(short_run)
        self.eps = eps

    @staticmethod
    def _student_t_log_prob(value, mean, kappa, alpha, beta, eps):
        """NIG posterior-predictive log density."""
        degrees = 2.0 * alpha
        scale_sq = beta * (kappa + 1.0) / (alpha * kappa).clamp_min(eps)
        z_sq = (value - mean).square() / scale_sq.clamp_min(eps)
        return (
            torch.lgamma((degrees + 1.0) / 2.0)
            - torch.lgamma(degrees / 2.0)
            - 0.5 * (torch.log(degrees) + torch.log(scale_sq) + value.new_tensor(torch.pi).log())
            - 0.5 * (degrees + 1.0) * torch.log1p(z_sq / degrees)
        )

    @staticmethod
    def _nig_update(value, mean, kappa, alpha, beta):
        next_kappa = kappa + 1.0
        delta = value - mean
        next_mean = mean + delta / next_kappa
        next_alpha = alpha + 0.5
        next_beta = beta + 0.5 * kappa * delta.square() / next_kappa
        return next_mean, next_kappa, next_alpha, next_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, steps, _, _ = x.shape
        if steps < 3:
            raise ValueError("Bayesian ICD requires a window of at least 3 time steps")

        batch, _, nodes, features = x.shape
        shape = (batch, 1, nodes, features)
        mean = x.new_zeros(shape)
        kappa = x.new_full(shape, self.prior_kappa)
        alpha = x.new_full(shape, self.prior_alpha)
        beta = x.new_full(shape, self.prior_beta)
        log_run = x.new_zeros(shape)
        log_hazard = x.new_tensor(self.hazard).log()
        log_growth = x.new_tensor(1.0 - self.hazard).log()
        evidence = []

        for time in range(steps):
            value = x[:, time].unsqueeze(1)
            log_predictive = self._student_t_log_prob(
                value, mean, kappa, alpha, beta, self.eps
            )

            # BOCD recursion: r_t=0 (changepoint) or r_t=r_{t-1}+1
            # (growth), followed by normalisation over every run length.
            log_joint = log_run + log_predictive
            cp = torch.logsumexp(log_joint + log_hazard, dim=1, keepdim=True)
            growth = log_joint + log_growth
            next_log_run = torch.cat([cp, growth], dim=1)
            next_log_run = next_log_run - torch.logsumexp(next_log_run, dim=1, keepdim=True)

            run_probability = next_log_run.exp()
            short_mass = run_probability[:, : min(self.short_run + 1, time + 2)].sum(dim=1)
            predictive_mean = (log_run.exp() * mean).sum(dim=1)
            predictive_var = (
                log_run.exp() * beta * (kappa + 1.0) / (alpha * kappa).clamp_min(self.eps)
            ).sum(dim=1)
            residual = (x[:, time] - predictive_mean) / predictive_var.clamp_min(self.eps).sqrt()
            valid = float(time >= 2)
            evidence.append(residual * short_mass * valid)

            # r_t=0 starts from the prior; growth hypotheses inherit their
            # respective sufficient statistics.
            prior_mean = x.new_zeros(shape)
            prior_kappa = x.new_full(shape, self.prior_kappa)
            prior_alpha = x.new_full(shape, self.prior_alpha)
            prior_beta = x.new_full(shape, self.prior_beta)
            reset = self._nig_update(
                value, prior_mean, prior_kappa, prior_alpha, prior_beta
            )
            grown = self._nig_update(value, mean, kappa, alpha, beta)
            mean = torch.cat([reset[0], grown[0]], dim=1)
            kappa = torch.cat([reset[1], grown[1]], dim=1)
            alpha = torch.cat([reset[2], grown[2]], dim=1)
            beta = torch.cat([reset[3], grown[3]], dim=1)
            log_run = next_log_run

        signed_evidence = torch.stack(evidence, dim=1)
        return self.encode(torch.relu(signed_evidence), torch.relu(-signed_evidence))


class DiffusionScoreMatchingBOCDBranch(_EvidenceEncoder):
    """Full Dm-BOCD ICD with the conjugate DSM Gaussian-mean model.

    This is the batched PyTorch counterpart of ``DSMGaussianUnknownMean`` in
    ``baselines/DSM-bocd-main``.  For the robustly standardised node features
    we use a Gaussian observation model with known unit variance.  The
    diffusion and its derivative follow the authors' experiments:

        m(x) = (1 + x^2)^(-1/2)
        m'(x) = -x (1 + x^2)^(-3/2)

    Every run-length hypothesis owns a Dm-posterior over its segment mean. Its
    precision and mean are updated by the diffusion score-matching quantities
    A(x) and v(x), then embedded in the exact BOCD hazard recursion.  ``omega``
    is positive and trainable, which adapts the generalised-Bayes learning rate
    using training data only instead of fixing a dataset-specific paper value.
    """

    def __init__(
        self,
        num_nodes,
        node_dim,
        hidden_dim,
        num_causes,
        hazard=1.0 / 50.0,
        omega=0.1,
        prior_mean=0.0,
        prior_variance=1.0,
        observation_variance=1.0,
        short_run=2,
        eps=1e-6,
    ):
        super().__init__(num_nodes, node_dim, hidden_dim, num_causes)
        self.hazard = float(hazard)
        self.prior_mean = float(prior_mean)
        self.prior_precision = float(1.0 / prior_variance)
        self.observation_variance = float(observation_variance)
        self.short_run = int(short_run)
        self.eps = eps
        # inverse softplus makes softplus(log_omega) start exactly at omega
        self.log_omega = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(omega)))))

    @property
    def omega(self):
        return torch.nn.functional.softplus(self.log_omega).clamp_min(self.eps)

    def _dsm_update(self, value, mean, precision):
        """Closed-form Dm-posterior update from Proposition 3.1."""
        variance = self.observation_variance
        m = torch.rsqrt(1.0 + value.square())
        grad_m = -value * (1.0 + value.square()).pow(-1.5)
        grad_r = 1.0 / variance
        grad_b = -value / variance
        a_value = grad_r * grad_r * m.square()
        # v = grad(r)^T m^2 grad(b) + div(mm^T)^T grad(r)
        v_value = grad_r * m.square() * grad_b + 2.0 * m * grad_m * grad_r
        next_precision = precision + 2.0 * self.omega * a_value
        next_mean = (precision * mean - 2.0 * self.omega * v_value) / next_precision
        return next_mean, next_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, features = x.shape
        if steps < 3:
            raise ValueError("Dm-BOCD ICD requires a window of at least 3 time steps")

        shape = (batch, 1, nodes, features)
        mean = x.new_full(shape, self.prior_mean)
        precision = x.new_full(shape, self.prior_precision)
        log_run = x.new_zeros(shape)
        log_hazard = x.new_tensor(self.hazard).log()
        log_growth = x.new_tensor(1.0 - self.hazard).log()
        evidence = []

        for time in range(steps):
            value = x[:, time].unsqueeze(1)
            predictive_variance = precision.reciprocal() + self.observation_variance
            log_predictive = -0.5 * (
                x.new_tensor(2.0 * torch.pi).log()
                + predictive_variance.log()
                + (value - mean).square() / predictive_variance
            )

            # Exact Adams-MacKay run-length recursion, using the robust
            # diffusion-score-matching posterior predictive above.
            log_joint = log_run + log_predictive
            cp = torch.logsumexp(log_joint + log_hazard, dim=1, keepdim=True)
            growth = log_joint + log_growth
            next_log_run = torch.cat([cp, growth], dim=1)
            next_log_run = next_log_run - torch.logsumexp(
                next_log_run, dim=1, keepdim=True
            )

            probability = next_log_run.exp()
            short_mass = probability[:, : min(self.short_run + 1, time + 2)].sum(dim=1)
            old_probability = log_run.exp()
            mixture_mean = (old_probability * mean).sum(dim=1)
            mixture_variance = (old_probability * predictive_variance).sum(dim=1)
            residual = (x[:, time] - mixture_mean) / mixture_variance.clamp_min(self.eps).sqrt()
            evidence.append(residual * short_mass * float(time >= 2))

            prior_mean = x.new_full(shape, self.prior_mean)
            prior_precision = x.new_full(shape, self.prior_precision)
            reset_mean, reset_precision = self._dsm_update(value, prior_mean, prior_precision)
            grown_mean, grown_precision = self._dsm_update(value, mean, precision)
            mean = torch.cat([reset_mean, grown_mean], dim=1)
            precision = torch.cat([reset_precision, grown_precision], dim=1)
            log_run = next_log_run

        signed_evidence = torch.stack(evidence, dim=1)
        return self.encode(torch.relu(signed_evidence), torch.relu(-signed_evidence))


def build_icd(
    method: str,
    num_nodes: int,
    node_dim: int,
    hidden_dim: int,
    num_causes: int,
    hazard: float = 1.0 / 50.0,
):
    """Construct an ICD branch without changing the surrounding architecture.

    ``hazard`` is exposed so that the expected BOCD run length can be examined
    in a controlled sensitivity experiment.  The default remains 1/50 and is
    therefore backward compatible with all existing training commands.
    """
    if method == "mad":
        return RobustEvidenceBranch(num_nodes, node_dim, hidden_dim, num_causes)
    if method in {"bocd", "bayesian"}:
        return BayesianOnlineEvidenceBranch(
            num_nodes, node_dim, hidden_dim, num_causes, hazard=hazard
        )
    if method == "dm_bocd":
        return DiffusionScoreMatchingBOCDBranch(
            num_nodes, node_dim, hidden_dim, num_causes, hazard=hazard
        )
    raise ValueError(f"Unsupported ICD method: {method}")
