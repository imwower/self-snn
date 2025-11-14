import argparse
from pathlib import Path

import torch
import yaml

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.core.pacemaker import PacemakerConfig
from self_snn.core.salience import SalienceConfig
from self_snn.core.wm import WorkingMemoryConfig
from self_snn.core.introspect import MetaConfig
from self_snn.memory.delay_mem import DelayMemoryConfig
from self_snn.worldmodel.pred_code import PredictiveCoderConfig
from self_snn.router.router import RouterConfig
from self_snn.agency.self_model import SelfModelConfig
from self_snn.agency.intention import IntentionConfig
from self_snn.agency.commit import CommitConfig
from self_snn.agency.act import ActConfig
from self_snn.agency.consistency import ConsistencyConfig
from self_snn.utils.logging_cn import setup_logger, log_pacemaker, log_self


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_self_snn_config(cfg: dict) -> SelfSNNConfig:
    backend = cfg.get("backend", {})
    mcc = cfg.get("mcc", {})
    memory = cfg.get("memory", {})
    router_cfg = cfg.get("router", {})
    agency_cfg = cfg.get("agency", {})
    runtime = cfg.get("runtime", {})

    pmc_cfg = mcc.get("pmc", {})
    sn_cfg = mcc.get("sn", {})
    wm_cfg = mcc.get("wm", {})
    mm_cfg = mcc.get("mm", {})

    delay_cfg = memory.get("delay", {})

    experts_cfg = router_cfg.get("experts", {})
    balance_cfg = router_cfg.get("balance", {})

    prospect_cfg = agency_cfg.get("prospect", {})
    weights_cfg = prospect_cfg.get("weights", {})
    commit_cfg = agency_cfg.get("commit", {})
    act_cfg = agency_cfg.get("act", {})

    dt_ms = runtime.get("dt_ms", 1.0)

    pmc = PacemakerConfig(
        n_neurons=pmc_cfg.get("N", 64),
        ring_delay_ms=pmc_cfg.get("ring_delay_ms", 8.0),
        ou_sigma=pmc_cfg.get("ou_sigma", 0.6),
        target_rate_hz=pmc_cfg.get("target_rate_hz", 3.0),
        dt_ms=dt_ms,
    )
    sal = SalienceConfig(
        err_tau_ms=sn_cfg.get("err_tau_ms", 50.0),
        gain_max=sn_cfg.get("gain_max", 2.0),
    )
    wm = WorkingMemoryConfig(
        microcolumns=wm_cfg.get("microcolumns", 3),
        neurons_per_uCol=tuple(wm_cfg.get("neurons_per_uCol", [80, 20])),
        stf_tau_ms=wm_cfg.get("stf_tau_ms", 200.0),
    )
    meta = MetaConfig(
        conf_window_ms=mm_cfg.get("conf_window_ms", 300.0),
        temp_base=mm_cfg.get("temp_base", 0.7),
    )
    delay = DelayMemoryConfig(
        dmax=delay_cfg.get("dmax", 120),
        dt_ms=delay_cfg.get("dt_ms", 1.0),
        eta_d=delay_cfg.get("eta_d", 0.1),
        delta_step=delay_cfg.get("delta_step", 1),
        delay_entropy_reg=delay_cfg.get("delay_entropy_reg", 1e-4),
    )
    pred = PredictiveCoderConfig(hidden_dim=128)

    router = RouterConfig(
        num_experts=experts_cfg.get("M", 16),
        k=experts_cfg.get("K", 2),
        z_loss=balance_cfg.get("z_loss", 1e-3),
    )

    self_model = SelfModelConfig(key_dim=agency_cfg.get("self_key_dim", 64))

    intention = IntentionConfig(
        n_candidates=prospect_cfg.get("n_candidates", 5),
        horizon=prospect_cfg.get("horizon", 10),
        w_lp=weights_cfg.get("lp", 0.45),
        w_empower=weights_cfg.get("empower", 0.2),
        w_reward=weights_cfg.get("reward", 0.15),
        c_energy=weights_cfg.get("energy", 0.1),
        c_risk=weights_cfg.get("risk", 0.07),
        c_boredom=weights_cfg.get("boredom", 0.03),
    )

    commit = CommitConfig(
        threshold=commit_cfg.get("threshold", 0.15),
        min_interval_s=commit_cfg.get("min_interval_s", 3.0),
        energy_cap=commit_cfg.get("energy_cap", 2.0e6),
        risk_cap=commit_cfg.get("risk_cap", 0.6),
    )

    act = ActConfig(
        router_topk=act_cfg.get("router_topk", 2),
        mpc_horizon=act_cfg.get("mpc_horizon", 5),
        efference_gain=act_cfg.get("efference_gain", 0.7),
    )

    consistency = ConsistencyConfig()

    return SelfSNNConfig(
        backend_engine=backend.get("engine", "torch-spkj"),
        device=backend.get("device", "cpu"),
        dt_ms=dt_ms,
        pmc=pmc,
        salience=sal,
        wm=wm,
        meta=meta,
        delay=delay,
        pred=pred,
        router=router,
        self_model=self_model,
        intention=intention,
        commit=commit,
        act=act,
        consistency=consistency,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--duration", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = build_self_snn_config(cfg)

    device = torch.device(model_cfg.device)
    model = SelfSNN(model_cfg).to(device)

    # 优先使用命令行 logdir，其次配置文件 logging.logdir，最后默认 runs/debug
    logdir = args.logdir or cfg.get("logging", {}).get("logdir", "runs/debug")
    logger = setup_logger(logdir)

    # 优先使用命令行 duration，其次 runtime.duration_s
    duration = args.duration or int(cfg.get("runtime", {}).get("duration_s", 180))

    task = cfg.get("task", "unspecified")
    logger.info(f"启动训练 task={task}, duration={duration}")

    for epoch in range(3):
        out = model(steps=duration)
        log_pacemaker(logger, "θ/γ 节律=..., 噪声σ=... 已启动")
        log_self(logger, model.self_model.report())
        loss = out["prediction_error"].abs().mean()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.data -= 1e-4 * p.grad
                p.grad.zero_()


if __name__ == "__main__":
    main()
