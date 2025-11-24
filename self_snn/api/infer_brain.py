from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict
from pathlib import Path

import torch
import yaml

from self_snn.core.workspace import SelfSNN, SelfSNNConfig


@dataclass
class BrainInput:
    """
    外部可传入的任务状态抽象。
    """

    task_id: str
    text_summary: str = ""
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class BrainSnapshot:
    """
    self-snn 短暂运行后的脑状态摘要。
    """

    region_activity: Dict[str, float]
    global_metrics: Dict[str, float]
    memory_summary: Dict[str, Any]
    decision_hint: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


def _load_config(cfg_path: str | Path) -> Dict[str, Any]:
    path = Path(cfg_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_model_config(cfg: Dict[str, Any], device: str | None) -> SelfSNNConfig:
    """
    轻量从 YAML 中读取关键信息覆盖默认 SelfSNNConfig。
    """
    model_cfg = SelfSNNConfig()

    backend = cfg.get("backend", {})
    model_cfg.backend_engine = backend.get("engine", model_cfg.backend_engine)
    model_cfg.device = device or backend.get("device", model_cfg.device)
    runtime = cfg.get("runtime", {})
    model_cfg.dt_ms = float(runtime.get("dt_ms", model_cfg.dt_ms))

    mcc_cfg = cfg.get("mcc", {})
    pmc_cfg = mcc_cfg.get("pmc", {})
    model_cfg.pmc.n_neurons = int(pmc_cfg.get("N", model_cfg.pmc.n_neurons))
    model_cfg.pmc.dt_ms = float(pmc_cfg.get("dt_ms", model_cfg.pmc.dt_ms))
    model_cfg.pmc.ring_delay_ms = float(pmc_cfg.get("ring_delay_ms", model_cfg.pmc.ring_delay_ms))
    model_cfg.pmc.ou_sigma = float(pmc_cfg.get("ou_sigma", model_cfg.pmc.ou_sigma))
    model_cfg.pmc.target_rate_hz = float(pmc_cfg.get("target_rate_hz", model_cfg.pmc.target_rate_hz))

    sal_cfg = mcc_cfg.get("sn", {})
    model_cfg.salience.err_tau_ms = float(sal_cfg.get("err_tau_ms", model_cfg.salience.err_tau_ms))
    model_cfg.salience.gain_max = float(sal_cfg.get("gain_max", model_cfg.salience.gain_max))

    gw_cfg = mcc_cfg.get("gw", {})
    model_cfg.gw.hubs = int(gw_cfg.get("hubs", model_cfg.gw.hubs))
    model_cfg.gw.ignite_thr = float(gw_cfg.get("ignite_thr", model_cfg.gw.ignite_thr))
    model_cfg.gw.nmda_tau_ms = float(gw_cfg.get("nmda_tau_ms", model_cfg.gw.nmda_tau_ms))
    model_cfg.gw.wta_inh = float(gw_cfg.get("wta_inh", model_cfg.gw.wta_inh))
    model_cfg.gw.k_select = int(gw_cfg.get("k_select", model_cfg.gw.k_select))

    wm_cfg = mcc_cfg.get("wm", {})
    if "microcolumns" in wm_cfg:
        model_cfg.wm.microcolumns = int(wm_cfg["microcolumns"])
    if "neurons_per_uCol" in wm_cfg:
        vals = wm_cfg["neurons_per_uCol"]
        if isinstance(vals, (list, tuple)) and len(vals) >= 2:
            model_cfg.wm.neurons_per_uCol = (int(vals[0]), int(vals[1]))
    if "stf_tau_ms" in wm_cfg:
        model_cfg.wm.stf_tau_ms = float(wm_cfg["stf_tau_ms"])

    memory_cfg = cfg.get("memory", {}).get("delay", {})
    model_cfg.delay.dmax = int(memory_cfg.get("dmax", model_cfg.delay.dmax))
    model_cfg.delay.dt_ms = float(memory_cfg.get("dt_ms", model_cfg.delay.dt_ms))

    router_cfg = cfg.get("router", {})
    exp_cfg = router_cfg.get("experts", {})
    model_cfg.router.num_experts = int(exp_cfg.get("M", model_cfg.router.num_experts))
    model_cfg.router.k = int(exp_cfg.get("K", model_cfg.router.k))
    bal_cfg = router_cfg.get("balance", {})
    model_cfg.router.z_loss = float(bal_cfg.get("z_loss", model_cfg.router.z_loss))
    model_cfg.router.usage_ema_tau = float(bal_cfg.get("usage_ema_tau", model_cfg.router.usage_ema_tau))

    agency_cfg = cfg.get("agency", {})
    model_cfg.self_model.key_dim = int(agency_cfg.get("self_key_dim", model_cfg.self_model.key_dim))
    prospect_cfg = agency_cfg.get("prospect", {})
    model_cfg.intention.n_candidates = int(prospect_cfg.get("n_candidates", model_cfg.intention.n_candidates))
    model_cfg.intention.horizon = int(prospect_cfg.get("horizon", model_cfg.intention.horizon))

    return model_cfg


def _build_drive(brain_input: BrainInput, steps: int, n_neurons: int) -> torch.Tensor:
    """
    将外部 features 简单映射为外部刺激，幅值与特征和相关。
    """
    magnitude = sum(abs(float(v)) for v in brain_input.features.values())
    # 控制在 [0, 1]，避免过强刺激
    mag = max(0.0, min(magnitude, 1.0))
    if mag <= 0.0 or steps <= 0 or n_neurons <= 0:
        return torch.zeros(steps, n_neurons)
    noise = torch.rand(steps, n_neurons)
    drive = (noise < mag).to(torch.float32)
    return drive


def _decide_mode(ignition_rate: float, kappa: float, mem_n_keys: float) -> Dict[str, Any]:
    """
    基于简单规则给出决策提示。
    """
    mode = "explore"
    confidence = 0.5
    if 0.8 <= kappa <= 1.2 and mem_n_keys > 0:
        mode = "exploit"
        confidence = 0.65
    elif kappa > 1.2 and mem_n_keys <= 0:
        mode = "chaotic"
        confidence = 0.4
    elif kappa < 0.8:
        mode = "stabilize"
        confidence = 0.55
    return {"mode": mode, "confidence": confidence, "kappa": kappa, "ignition_rate": ignition_rate}


@torch.no_grad()
def run_brain_step(
    cfg_path: str,
    brain_input: BrainInput,
    steps: int = 10,
    device: str | None = None,
) -> BrainSnapshot:
    """
    读取 config，运行 self-snn 若干步，并返回脑状态摘要。
    """
    cfg = _load_config(cfg_path)
    model_cfg = _build_model_config(cfg, device)
    model = SelfSNN(model_cfg)

    drive = _build_drive(brain_input, steps, model_cfg.pmc.n_neurons)
    drive_input = {"drive": drive} if drive.numel() > 0 else None
    out = model(drive_input, steps=steps)

    spikes = out.get("spikes")
    mean_rate = float(spikes.float().mean() * 1000.0 / max(model_cfg.dt_ms, 1e-6)) if spikes is not None else 0.0
    ignition_rate = float(out.get("ignition_rate", 0.0))
    branching_kappa = float(out.get("branching_kappa", 0.0))
    moe_ratio = float(out.get("moe_energy_ratio", 0.0))

    wm_state = out.get("wm_state")
    router_stats = out.get("router_stats", {})
    commit_state = out.get("commit_state", {})
    act_out = out.get("act_out", {})

    region_activity = {
        "mcc": mean_rate,
        "wm": float(wm_state.mean().item()) if wm_state is not None else 0.0,
        "router": float(router_stats.get("probs", torch.tensor(0.0)).mean().item())
        if router_stats
        else 0.0,
        "agency": float(act_out.get("energy", 0.0)),
    }

    mem_stats = model.memory.stats()
    memory_summary = {
        "n_keys": mem_stats.get("n_keys", 0.0),
        "mean_delay": mem_stats.get("mean_delay", 0.0),
        "var_delay": mem_stats.get("var_delay", 0.0),
    }

    global_metrics = {
        "mean_spike_rate_hz": mean_rate,
        "ignition_rate": ignition_rate,
        "branching_kappa": branching_kappa,
        "moe_energy_ratio": moe_ratio,
    }

    decision_hint = _decide_mode(ignition_rate, branching_kappa, mem_stats.get("n_keys", 0.0))

    return BrainSnapshot(
        region_activity=region_activity,
        global_metrics=global_metrics,
        memory_summary=memory_summary,
        decision_hint=decision_hint,
    )


def snapshot_to_dict(snapshot: BrainSnapshot) -> Dict[str, Any]:
    return {
        "region_activity": snapshot.region_activity,
        "global_metrics": snapshot.global_metrics,
        "memory_summary": snapshot.memory_summary,
        "decision_hint": snapshot.decision_hint,
        "created_at": snapshot.created_at,
    }

