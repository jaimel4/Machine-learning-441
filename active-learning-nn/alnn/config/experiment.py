from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any

TaskType = Literal["classification", "regression"]

@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    activation: Literal["relu", "tanh"] = "relu"
    dropout: float = 0.0

@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-2
    weight_decay: float = 1e-4
    early_stopping_patience: int = 20

@dataclass
class ActiveLearningConfig:
    strategy: Literal["passive", "uncertainty", "sensitivity"] = "passive"
    init_frac: float = 0.1
    query_size: int = 5
    budget: int = 200
    mc_dropout_passes: int = 0  # >0 enables MC-dropout uncertainty

@dataclass
class ExperimentConfig:
    name: str
    task: TaskType
    dataset: str
    seed: int = 42
    repeats: int = 5
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    al: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    extras: Dict[str, Any] = field(default_factory=dict)
