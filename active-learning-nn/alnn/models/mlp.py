from __future__ import annotations
import torch
import torch.nn as nn

def _act(name: str):
    return nn.ReLU() if name == "relu" else nn.Tanh()

class MLP1H(nn.Module):
    """Single-hidden-layer MLP for both classification and regression."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _act(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
