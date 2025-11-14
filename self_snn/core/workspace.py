from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

import torch
import torch.nn as nn

from .pacemaker import Pacemaker, PacemakerConfig
from .salience import SalienceModule, SalienceConfig
from .wm import WorkingMemory, WorkingMemoryConfig
from .introspect import MetaIntrospector, MetaConfig
from ..memory.delay_mem import DelayMemory, DelayMemoryConfig
from ..worldmodel.pred_code import PredictiveCoder, PredictiveCoderConfig
from ..router.router import GWRouter, RouterConfig
from ..agency.self_model import SelfModel, SelfModelConfig
from ..agency.intention import IntentionModule, IntentionConfig
from ..agency.commit import CommitModule, CommitConfig
from ..agency.act import ActModule, ActConfig
from ..agency.consistency import ConsistencyModule, ConsistencyConfig


@dataclass
class SelfSNNConfig:
    backend_engine: str = "torch-spkj"
    device: str = "cpu"
    dt_ms: float = 1.0
    pmc: PacemakerConfig = field(default_factory=PacemakerConfig)
    salience: SalienceConfig = field(default_factory=SalienceConfig)
    wm: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    delay: DelayMemoryConfig = field(default_factory=DelayMemoryConfig)
    pred: PredictiveCoderConfig = field(default_factory=PredictiveCoderConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    self_model: SelfModelConfig = field(default_factory=SelfModelConfig)
    intention: IntentionConfig = field(default_factory=IntentionConfig)
    commit: CommitConfig = field(default_factory=CommitConfig)
    act: ActConfig = field(default_factory=ActConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)


class SelfSNN(nn.Module):
    def __init__(self, config: SelfSNNConfig) -> None:
        super().__init__()
        device = torch.device(config.device)
        self.config = config

        self.pacemaker = Pacemaker(config.pmc, device=device)
        self.salience = SalienceModule(config.salience)
        self.workspace_wm = WorkingMemory(config.wm)
        self.meta = MetaIntrospector(config.meta)
        self.memory = DelayMemory(config.delay)
        self.pred = PredictiveCoder(config.pred)
        self.router = GWRouter(config.router)

        self.self_model = SelfModel(config.self_model)
        self.intention = IntentionModule(config.intention)
        self.commit = CommitModule(config.commit)
        self.act = ActModule(config.act)
        self.consistency = ConsistencyModule(config.consistency)

    def forward(self, inputs: Dict[str, torch.Tensor] | None = None, steps: int = 100) -> Dict[str, Any]:
        device = next(self.parameters()).device
        inputs = inputs or {}
        ext_drive = inputs.get("drive")

        spikes = self.pacemaker(T=steps).to(device)
        if ext_drive is not None:
            ext_drive = ext_drive.to(device)
            L = min(ext_drive.shape[0], spikes.shape[0])
            if ext_drive.dim() == 1:
                drive = ext_drive.unsqueeze(0).expand(L, -1)
            else:
                drive = ext_drive[:L]
            if drive.shape[1] != spikes.shape[1]:
                drive = drive[:, : spikes.shape[1]]
            drive_spikes = (drive > 0).to(spikes.dtype)
            spikes[:L] = torch.clamp(spikes[:L] + drive_spikes, 0.0, 1.0)

        wm_state = self.workspace_wm(spikes)
        pred, pred_err = self.pred(wm_state)
        gain = self.salience(pred_err, dt_ms=self.config.dt_ms)
        meta = self.meta(pred_err, gain)

        self.memory.write(key=self.self_model.key, sequence=wm_state)

        gw_mask, router_stats = self.router(wm_state)
        goals, utilities = self.intention(self.memory, self.pred, self.self_model)
        commit_state = self.commit(goals, utilities, meta, self.self_model)
        act_out = self.act(commit_state, wm_state, gw_mask)
        credit = self.consistency(commit_state, act_out)

        self.self_model.update_state(meta, act_out)

        # v0: 自发 / 点火统计（简化版）
        spikes_f = spikes.float()
        spike_counts = spikes_f.sum(dim=1)
        n_neurons = spikes.shape[1]
        ignition_steps = spike_counts > 0.1 * float(n_neurons)
        ignition_rate = ignition_steps.float().mean()
        if spike_counts.numel() > 1:
            ratio = spike_counts[1:] / torch.clamp(spike_counts[:-1], min=1.0)
            branching_kappa = ratio.mean()
        else:
            branching_kappa = torch.tensor(1.0, device=device)

        # v1: 条件计算能耗（Top-K 专家占比）
        num_experts = gw_mask.numel()
        active_experts = gw_mask.sum()
        moe_energy_ratio = active_experts / max(float(num_experts), 1.0)

        return {
            "spikes": spikes,
            "wm_state": wm_state,
            "prediction": pred,
            "prediction_error": pred_err,
            "salience_gain": gain,
            "meta": meta,
            "router_mask": gw_mask,
            "router_stats": router_stats,
            "goals": goals,
            "utilities": utilities,
            "commit_state": commit_state,
            "act_out": act_out,
            "self_credit": credit,
            "ignition_rate": ignition_rate,
            "branching_kappa": branching_kappa,
            "moe_energy_ratio": moe_energy_ratio,
        }
