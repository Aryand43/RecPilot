"""Pydantic settings for RecPilot experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from recpilot.paths import DEFAULT_CONFIG, DEFAULT_DATA_DIR, REPO_ROOT


class Budget(BaseModel):
    max_iters: int = 12
    max_wall_s: float = 14400
    max_tokens: int = 200000
    train_timeout_s: float = 600
    keep_delta: float = 1e-4
    converge_eps: float = 0.002
    converge_n: int = 3
    exploration_min_iters: int = 10
    max_retries: int = 1
    cooldown_iters: int = 2
    regression_tol: float = 0.01


class ModelConfig(BaseModel):
    name: str = "fm"  # fm | bpr | listwise | multitask | sequence_interest | deepfm_din
    k: int = 16
    lr: float = 0.001
    l2: float = 1e-6
    epochs: int = 40
    batch_size: int = 8192
    patience: int = 4
    seed: int = 0
    blend_pop: float = 0.0
    listwise_temperature: float = 1.0
    aux_click_weight: float = 0.3
    aux_like_weight: float = 0.1
    seq_len: int = 20
    seq_half_life: float = 7.0
    play_weight: float = 0.05
    seq_engage_click: float = 0.0
    seq_engage_like: float = 0.0
    seq_engage_play: float = 0.0
    seq_listwise: bool = False
    seq_aux: bool = False


class FeatureConfig(BaseModel):
    use_kit_encode: bool = True
    history_crosses: bool = False
    time_features: bool = False
    recency_history: bool = False
    recency_variant: str = "hl7"  # hl2 | hl7 | last5


class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"


class Settings(BaseModel):
    data_dir: str = str(DEFAULT_DATA_DIR)
    runs_dir: str = str(REPO_ROOT / "runs")
    budget: Budget = Field(default_factory=Budget)
    model: ModelConfig = Field(default_factory=ModelConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    def resolved_data_dir(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    def resolved_runs_dir(self) -> Path:
        p = Path(self.runs_dir)
        return p if p.is_absolute() else REPO_ROOT / p


class ExperimentSpec(BaseModel):
    run_id: str
    hypothesis: str
    operator: str
    params: dict[str, Any] = Field(default_factory=dict)
    parent_run: Optional[str] = None
    config: Settings
    retry: int = 0


def load_settings(path: Optional[Path] = None) -> Settings:
    src = path or DEFAULT_CONFIG
    if src.exists():
        with open(src) as fh:
            raw = yaml.safe_load(fh) or {}
        return Settings.model_validate(raw)
    return Settings()
