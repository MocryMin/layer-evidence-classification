"""Probe head variants for EXP-20260729-002 diagnostics.

All heads end in a linear classifier ``768 -> 150`` with Xavier-uniform weight
and zero bias. They differ in the (learnable or fixed) feature transform applied
before the classifier:

- ``PlainHead``       : no transform (pure linear probe, ``W x + b``).
- ``LNHead``          : LayerNorm ``(x-mu)/sigma * gamma + beta`` (full LN).
- ``NormOnlyHead``    : ``(x-mu)/sigma`` with fixed gamma=1, beta=0 (ablation:
                        normalization without affine).
- ``AffineOnlyHead``  : ``gamma * x + beta`` (ablation: affine without
                        normalization). Note: affine+linear is a reparameterised
                        linear classifier, so this isolates optimisation, not
                        expressiveness.
"""
from __future__ import annotations

import torch
import torch.nn as nn

HEAD_TYPES = ["plain", "ln", "norm_only", "affine_only"]


def _init_linear(fc: nn.Linear) -> None:
    nn.init.xavier_uniform_(fc.weight)
    nn.init.zeros_(fc.bias)


class PlainHead(nn.Module):
    def __init__(self, in_dim: int = 768, n_classes: int = 150):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
        _init_linear(self.fc)

    def forward(self, x):
        return self.fc(x)


class LNHead(nn.Module):
    def __init__(self, in_dim: int = 768, n_classes: int = 150):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, n_classes)
        _init_linear(self.fc)

    def forward(self, x):
        return self.fc(self.ln(x))


class NormOnlyHead(nn.Module):
    """LayerNorm with elementwise_affine=False (pure normalization, no gamma/beta)."""

    def __init__(self, in_dim: int = 768, n_classes: int = 150):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.fc = nn.Linear(in_dim, n_classes)
        _init_linear(self.fc)

    def forward(self, x):
        return self.fc(self.ln(x))


class AffineOnlyHead(nn.Module):
    def __init__(self, in_dim: int = 768, n_classes: int = 150):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(in_dim))
        self.beta = nn.Parameter(torch.zeros(in_dim))
        self.fc = nn.Linear(in_dim, n_classes)
        _init_linear(self.fc)

    def forward(self, x):
        return self.fc(self.gamma * x + self.beta)


class MLPHead(nn.Module):
    """Single-hidden-layer MLP probe (nonlinear-capacity control, task 04).

    ``Linear(768, r, bias=True) + ReLU + Linear(r, 150, bias=False)``.
    Parameter count = 768r (W1) + r (b1) + 150r (W2) = 919r; r=128 gives
    117,632 params, matching the plain ``linear_with_bias`` head's 115,350
    within +2% (the closest integer r; exact match would be r=125.52).
    Initialisation follows the other heads: Xavier-uniform weights, zero bias.
    """

    def __init__(self, in_dim: int = 768, n_classes: int = 150, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, n_classes, bias=False)
        _init_linear(self.fc1)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ActHead(nn.Module):
    """MLP probe with configurable activation (activation ablation, task 05).

    ``z = W2 * act(W1 x + b1) + b2`` - both layers carry a bias (919r + 150
    params at hidden=r). ``act`` in {``none``, ``relu``, ``leaky``, ``gelu``,
    ``sigmoid``} (LeakyReLU negative_slope=0.01, GELU exact, Sigmoid bounded
    saturating). Used to test whether the ReLU dead-unit lock-in is specific
    to ReLU: ``none`` cannot dead-lock at all (pure 2-layer linear),
    ``leaky``/``gelu`` keep a gradient through the negative interval,
    ``sigmoid`` is the bounded-saturating control (output always positive,
    gradient -> 0 in saturation). Initialisation follows the other heads.
    """

    def __init__(self, in_dim: int = 768, n_classes: int = 150, hidden: int = 128,
                 act: str = "none"):
        super().__init__()
        self.act_name = act
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, n_classes, bias=True)
        _init_linear(self.fc1)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        if act == "none":
            self.act = nn.Identity()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "leaky":
            self.act = nn.LeakyReLU(negative_slope=0.01)
        elif act == "gelu":
            self.act = nn.GELU()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError(f"unknown activation {act}")

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def build_head(head_type: str, in_dim: int = 768, n_classes: int = 150) -> nn.Module:
    cls = {
        "plain": PlainHead,
        "ln": LNHead,
        "norm_only": NormOnlyHead,
        "affine_only": AffineOnlyHead,
        "mlp": MLPHead,
        "act": ActHead,
    }[head_type]
    return cls(in_dim, n_classes)
