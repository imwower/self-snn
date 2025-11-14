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
from ..router.experts import MaskedExperts, ExpertsConfig
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
    experts: ExpertsConfig = field(default_factory=ExpertsConfig)
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

        # 将 WM 状态映射到 MoE / 世界模型的隐空间
        self.wm_to_hidden = nn.Linear(1, config.pred.hidden_dim)
        # GW-MoE 专家网络：条件计算
        experts_cfg = ExpertsConfig(
            input_dim=config.pred.hidden_dim,
            output_dim=config.pred.hidden_dim,
            num_experts=config.router.num_experts,
        )
        self.experts = MaskedExperts(experts_cfg)

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
        # 将 WM 状态映射到 MoE / 世界模型隐藏空间
        hidden = self.wm_to_hidden(wm_state.unsqueeze(0)).squeeze(0)

        # GW-MoE: 条件专家选择
        gw_mask, router_stats = self.router(hidden)
        # 条件计算：仅对被选中的专家执行前向
        expert_out, synops = self.experts(hidden, gw_mask)

        # 世界模型在专家输出的隐藏表示上工作
        pred, pred_err = self.pred(expert_out)

        # Salience / Meta 内省
        gain = self.salience(pred_err, dt_ms=self.config.dt_ms)
        meta = self.meta(pred_err, gain)

        # D-MEM: 基于 spike 统计与第三因子（这里用 confidence - uncertainty 近似）
        spike_counts = spikes.float().sum(dim=1)
        third_factor = float((meta["confidence"] - meta["uncertainty"]).detach())
        self.memory.update_delay_from_spikes(self.self_model.key, spike_counts, third_factor=third_factor)
        self.memory.write(key=self.self_model.key, sequence=wm_state)

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

        # v1: 条件计算能耗（真实 MoE synops 对比密集前向）
        num_experts = len(self.experts.experts)
        synops_masked = float(synops.sum())
        per_expert_cost = float(hidden.numel() * hidden.numel())
        synops_dense = per_expert_cost * num_experts
        moe_energy_ratio = synops_masked / max(synops_dense, 1.0)

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
