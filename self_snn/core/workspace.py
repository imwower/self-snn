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
from ..worldmodel.imagination import ImaginationEngine, ImaginationConfig
from ..router.router import GWRouter, RouterConfig
from ..router.experts import MaskedExperts, ExpertsConfig
from ..agency.self_model import SelfModel, SelfModelConfig
from ..agency.intention import IntentionModule, IntentionConfig
from ..agency.commit import CommitModule, CommitConfig
from ..agency.act import ActModule, ActConfig
from ..agency.consistency import ConsistencyModule, ConsistencyConfig


@dataclass
class GlobalWorkspaceConfig:
    """
    Global Workspace（GW）配置。

    Attributes
    ----------
    hubs:
        GW 中的“汇聚节点”数量（通常与皮层热点类似数量级）。
    ignite_thr:
        NMDA 慢变量的点火阈值，超过视为进入点火状态。
    nmda_tau_ms:
        NMDA/α 函数时间常数（ms），控制慢突触保持时间。
    wta_inh:
        软 k-WTA 抑制系数，越大越接近硬 WTA。
    k_select:
        每次点火时选取的 Top-K hub 数量。
    """

    hubs: int = 160
    ignite_thr: float = 0.58
    nmda_tau_ms: float = 100.0
    wta_inh: float = 0.85
    k_select: int = 2


class GlobalWorkspace(nn.Module):
    """
    Global Workspace：软 k-WTA + NMDA 慢突触 + 点火检测与广播。

    输入为起搏器的 spike 序列 [T, N]，内部维护 NMDA 风格的慢变量，
    在每个时间步上根据 NMDA 活动选择 Top-K hub 视作点火，并给出
    ignite_mask（[T, N]）与 coverage（每步点火覆盖度）。
    """

    def __init__(self, config: GlobalWorkspaceConfig, dt_ms: float, n_neurons: int, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.dt_ms = dt_ms
        self.n_neurons = n_neurons
        self.device = device
        # NMDA 风格慢突触状态
        self.register_buffer("nmda_state", torch.zeros(n_neurons, device=device))

    @torch.no_grad()
    def forward(self, spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        spikes:
            形状 [T, N] 的二值脉冲序列。
        """
        assert spikes.dim() == 2, "GlobalWorkspace 期望输入形状为 [T, N]"
        T, N = spikes.shape
        nmda = self.nmda_state

        alpha = self.dt_ms / max(self.config.nmda_tau_ms, 1.0)
        ignite_mask = torch.zeros_like(spikes, dtype=torch.float32)
        coverage = torch.zeros(T, device=spikes.device, dtype=torch.float32)

        for t in range(T):
            s_t = spikes[t].float()
            # NMDA 慢突触积分
            nmda = (1.0 - alpha) * nmda + alpha * s_t

            max_nmda = nmda.max()
            if max_nmda < self.config.ignite_thr:
                # 未达点火阈值，不触发 GW
                coverage[t] = 0.0
                continue

            k = min(self.config.k_select, N)
            if k > 0:
                # 软 k-WTA：按 NMDA 状态选 Top-K 作为点火 hub
                scores = nmda * self.config.wta_inh
                _, idx = torch.topk(scores, k=k)
                ignite_mask[t, idx] = 1.0
                coverage[t] = ignite_mask[t].mean()

        self.nmda_state = nmda.detach()
        return {"ignite_mask": ignite_mask, "coverage": coverage}


@dataclass
class SelfSNNConfig:
    backend_engine: str = "torch-spkj"
    device: str = "cpu"
    dt_ms: float = 1.0
    pmc: PacemakerConfig = field(default_factory=PacemakerConfig)
    salience: SalienceConfig = field(default_factory=SalienceConfig)
    gw: GlobalWorkspaceConfig = field(default_factory=GlobalWorkspaceConfig)
    wm: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    delay: DelayMemoryConfig = field(default_factory=DelayMemoryConfig)
    pred: PredictiveCoderConfig = field(default_factory=PredictiveCoderConfig)
    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
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
        self.global_workspace = GlobalWorkspace(config.gw, dt_ms=config.dt_ms, n_neurons=config.pmc.n_neurons, device=device)
        self.workspace_wm = WorkingMemory(config.wm)
        self.meta = MetaIntrospector(config.meta)
        self.memory = DelayMemory(config.delay)
        self.pred = PredictiveCoder(config.pred)
        self.imagination = ImaginationEngine(config.imagination)
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

        # 起搏器：生成自发脉冲、膜电位与相位信息
        pmc_out = self.pacemaker.step(steps)
        spikes = pmc_out["spikes"].to(device)
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

        # Global Workspace：基于 spike 活动的软 k-WTA 点火与覆盖度
        gw_out = self.global_workspace(spikes)
        ignite_mask = gw_out["ignite_mask"]
        ignite_coverage = gw_out["coverage"]

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

        goals, utilities = self.intention(self.memory, self.pred, self.self_model, self.imagination)
        commit_state = self.commit(goals, utilities, meta, self.self_model)
        act_out = self.act(commit_state, wm_state, gw_mask)
        credit = self.consistency(commit_state, act_out)

        self.self_model.update_state(meta, act_out)

        # v0: 自发 / 点火统计
        spikes_f = spikes.float()
        spike_counts = spikes_f.sum(dim=1)
        n_neurons = spikes.shape[1]
        # 若 GW 有点火，则使用 coverage>0 的比例作为点火率；否则退回 spike 阈值判断
        if ignite_coverage.numel() > 0 and ignite_coverage.max() > 0:
            ignition_rate = (ignite_coverage > 0).float().mean()
        else:
            ignition_steps = spike_counts > 0.1 * float(n_neurons)
            ignition_rate = ignition_steps.float().mean()

        branching_kappa = pmc_out["branching_kappa"].to(device)

        # v1: 条件计算能耗（真实 MoE synops 对比密集前向）
        num_experts = len(self.experts.experts)
        synops_masked = float(synops.sum())
        per_expert_cost = float(hidden.numel() * hidden.numel())
        synops_dense = per_expert_cost * num_experts
        moe_energy_ratio = synops_masked / max(synops_dense, 1.0)

        return {
            "spikes": spikes,
            "vm_trace": pmc_out["vm"],
            "theta_phase": pmc_out["theta_phase"],
            "gamma_phase": pmc_out["gamma_phase"],
            "ignite_mask": ignite_mask,
            "ignite_coverage": ignite_coverage,
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
