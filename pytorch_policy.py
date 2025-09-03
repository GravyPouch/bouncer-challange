from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


@dataclass
class ScenarioConfig:
    attribute_ids: List[str]
    min_counts: Dict[str, int]
    rel_freq: Dict[str, float]
    correlations: Dict[str, Dict[str, float]]
    capacity: int = 1000


class TinyPolicy(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        # Monotonic-leaning init: prefer positive influence from attributes
        with torch.no_grad():
            self.net[0].weight.data = self.net[0].weight.data.abs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def bernoulli_copula_sample(batch_size: int,
                            rel_freq: Dict[str, float],
                            correlations: Dict[str, Dict[str, float]],
                            attribute_ids: List[str]) -> torch.Tensor:
    """
    Very small synthetic sampler that approximates joint Bernoulli with given
    marginals and pairwise correlations via Gaussian copula.
    Returns tensor [B, D] of 0/1.
    """
    device = torch.device('cpu')
    D = len(attribute_ids)

    # Build correlation matrix in Gaussian space (clip to SPD-ish) using NumPy linalg for lint compatibility
    R = np.eye(D)
    for i, ai in enumerate(attribute_ids):
        for j, aj in enumerate(attribute_ids):
            if i == j:
                continue
            rho = float(correlations.get(ai, {}).get(aj, 0.0))
            R[i, j] = rho
            R[j, i] = rho
    # Regularize to ensure positive-definiteness (eigenvalue floor + small jitter)
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.clip(eigvals, a_min=1e-3, a_max=None)
    R = (eigvecs @ np.diag(eigvals) @ eigvecs.T)
    # Final jitter for numerical stability
    R = R + 1e-6 * np.eye(D)

    # Sample standard normals with covariance R
    L = np.linalg.cholesky(R)
    z = np.random.randn(batch_size, D) @ L
    z = torch.from_numpy(z.astype(np.float32))
    # Convert to uniform via CDF, then to Bernoulli via thresholds from marginals
    u = 0.5 * (1 + torch.erf(z / (2 ** 0.5)))
    thresholds = torch.tensor([rel_freq[a] for a in attribute_ids]).unsqueeze(0)
    x = (u < thresholds).float().to(device)
    return x


def train_policy(config: ScenarioConfig,
                 steps: int = 2000,
                 batch_size: int = 4096,
                 lr: float = 1e-3) -> Tuple[TinyPolicy, Dict[str, float]]:
    D = len(config.attribute_ids)
    policy = TinyPolicy(D)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Dual variables per-attribute for min constraints
    lambdas = {a: 0.0 for a in config.attribute_ids}
    min_counts_tensor = torch.tensor([config.min_counts.get(a, 0) for a in config.attribute_ids]).float()

    # Acceptance rate box [0.40, 0.75]
    acceptance_lower_bound = 0.40
    acceptance_upper_bound = 0.75

    for _ in range(steps):
        x = bernoulli_copula_sample(batch_size, config.rel_freq, config.correlations, config.attribute_ids)
        p = policy(x)

        # Expected accepted by attribute (per batch), scale to capacity
        accepted_by_attr = (p.unsqueeze(1) * x).mean(dim=0) * config.capacity

        violations = torch.clamp(min_counts_tensor - accepted_by_attr, min=0.0)
        # Enforce acceptance rate box via hinge-squared penalties
        p_mean = p.mean()
        over = torch.clamp(p_mean - acceptance_upper_bound, min=0.0)
        under = torch.clamp(acceptance_lower_bound - p_mean, min=0.0)
        accept_rate_penalty = over.pow(2) + under.pow(2)
        # Encourage high entropy to avoid p near 0/1
        eps = 1e-6
        entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)).mean()

        # Penalty weight from current lambdas
        lambda_vec = torch.tensor([lambdas[a] for a in config.attribute_ids])
        constraint_penalty = (lambda_vec * (violations / config.capacity)).sum()

        # Margin encouragement for extreme combos
        Dsum = x.sum(dim=1)
        all_one = (Dsum == x.size(1)).float()
        all_zero = (Dsum == 0).float()
        margin_up = torch.clamp(0.8 - p, min=0.0) * all_one
        margin_down = torch.clamp(p - 0.2, min=0.0) * all_zero

        loss = (
            constraint_penalty
            + 5.0 * accept_rate_penalty
            + 0.02 * (margin_up.mean() + margin_down.mean())
            - 0.01 * entropy
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Enforce non-negative first-layer weights after each step
        with torch.no_grad():
            policy.net[0].weight.data = policy.net[0].weight.data.abs()

        # Dual ascent (simple, stable update)
        for idx, a in enumerate(config.attribute_ids):
            # Smaller dual step to prevent over-conservative zeros
            lambdas[a] = max(0.0, lambdas[a] + 2.0 * (violations[idx].item() / max(1.0, config.capacity)))

    return policy, lambdas


class LinearBidPricePolicy(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)
        with torch.no_grad():
            # Encourage monotonicity: start with non-negative weights
            self.linear.weight.data = self.linear.weight.data.abs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)).squeeze(-1)


def train_bidprice_policy(config: ScenarioConfig,
                          steps: int = 1500,
                          batch_size: int = 4096,
                          lr: float = 5e-4) -> Tuple[LinearBidPricePolicy, Dict[str, float]]:
    """
    Train a simple linear bid-price style policy with dual updates.

    The model approximates accept prob p(x) = sigmoid(w^T x + b). We penalize expected
    rejections, enforce meeting min counts via dual multipliers, and keep the overall
    acceptance rate near a feasibility-driven target.
    """
    D = len(config.attribute_ids)
    policy = LinearBidPricePolicy(D)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    lambdas = {a: 0.0 for a in config.attribute_ids}
    min_counts_tensor = torch.tensor([config.min_counts.get(a, 0) for a in config.attribute_ids]).float()

    # Acceptance rate box [0.40, 0.75]
    acceptance_lower_bound = 0.40
    acceptance_upper_bound = 0.75

    for _ in range(steps):
        x = bernoulli_copula_sample(batch_size, config.rel_freq, config.correlations, config.attribute_ids)
        p = policy(x)

        # Expected accepted by attribute (per batch), scale to capacity
        accepted_by_attr = (p.unsqueeze(1) * x).mean(dim=0) * config.capacity

        # Slack-aware violations: add small buffer to fight variance in online fill
        buffer = 0.005 * config.capacity
        buffer = 0.005 * config.capacity
        violations = torch.clamp((min_counts_tensor + buffer) - accepted_by_attr, min=0.0)

        # Objectives and regularizers: box constraints on acceptance rate
        p_mean = p.mean()
        over = torch.clamp(p_mean - acceptance_upper_bound, min=0.0)
        under = torch.clamp(acceptance_lower_bound - p_mean, min=0.0)
        accept_rate_penalty = over.pow(2) + under.pow(2)
        eps = 1e-6
        entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)).mean()

        lambda_vec = torch.tensor([lambdas[a] for a in config.attribute_ids])
        constraint_penalty = (lambda_vec * (violations / config.capacity)).sum()

        loss = (
            constraint_penalty
            + 4.0 * accept_rate_penalty
            - 0.01 * entropy
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Enforce non-negative monotone weights
        with torch.no_grad():
            policy.linear.weight.data = policy.linear.weight.data.abs()

        # Conservative dual ascent
        for idx, a in enumerate(config.attribute_ids):
            lambdas[a] = max(0.0, lambdas[a] + 1.0 * (violations[idx].item() / max(1.0, config.capacity)))

    return policy, lambdas


def save_policy(policy: nn.Module, path: str):
    arch = "linear" if isinstance(policy, LinearBidPricePolicy) else "tiny"
    torch.save({"state_dict": policy.state_dict(), "arch": arch}, path)


def load_policy(path: str, num_features: int) -> nn.Module:
    payload = torch.load(path, map_location='cpu')
    # New format
    if isinstance(payload, dict) and "state_dict" in payload and "arch" in payload:
        arch = payload["arch"]
        state_dict = payload["state_dict"]
        if arch == "linear":
            model: nn.Module = LinearBidPricePolicy(num_features)
        else:
            model = TinyPolicy(num_features)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    # Backward compatibility: raw state_dict of TinyPolicy
    try:
        model_bc = TinyPolicy(num_features)
        model_bc.load_state_dict(payload)
        model_bc.eval()
        return model_bc
    except Exception:
        # Try linear as last resort
        model_lin = LinearBidPricePolicy(num_features)
        model_lin.load_state_dict(payload)
        model_lin.eval()
        return model_lin


def policy_decide(policy: TinyPolicy, attr_vector: Dict[str, bool], attribute_ids: List[str], threshold: float = 0.5) -> bool:
    x = torch.tensor([[1.0 if attr_vector.get(a, False) else 0.0 for a in attribute_ids]])
    with torch.no_grad():
        p = policy(x).item()
    return p >= threshold


