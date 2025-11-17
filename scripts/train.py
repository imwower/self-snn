import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

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
from self_snn.utils.encoders import encode_text, encode_image, encode_video
from self_snn.utils.logging_cn import (
    setup_logger,
    log_pacemaker,
    log_ignition,
    log_router,
    log_energy,
    log_self,
    log_self_think,
    log_self_want,
    log_self_do,
    log_memory_read,
)


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
        usage_ema_tau=balance_cfg.get("usage_ema_tau", 1000),
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


@torch.no_grad()
def compute_task_metrics(
    model: SelfSNN,
    task: str,
    writer: SummaryWriter,
    logger,
    epoch: int,
    steps_eval: int = 30,
    data_root: str = "data/toy",
) -> None:
    """
    基于 toy 数据集计算任务级指标（命中率/成功率），并写入 TensorBoard 与中文日志。
    """
    device = next(model.parameters()).device
    data_root_path = Path(data_root)

    if task == "words_images":
        path = data_root_path / "words_images.npz"
        if not path.exists():
            logger.warning(f"words_images toy data not found at {path}")
            return
        npz = np.load(path, allow_pickle=True)
        words = npz["words"]
        images = npz["images"]
        n = len(words)

        text_feats = []
        img_feats = []
        for i in range(n):
            tokens = [str(words[i])]
            text_vec = encode_text(tokens, dim=128)
            img_tensor = torch.from_numpy(images[i])
            img_vec = encode_image(img_tensor, dim=128)

            drive_text = text_vec.unsqueeze(0).repeat(steps_eval, 1).to(device)
            drive_img = img_vec.unsqueeze(0).repeat(steps_eval, 1).to(device)

            out_text = model({"drive": drive_text}, steps=steps_eval)
            out_img = model({"drive": drive_img}, steps=steps_eval)
            text_feats.append(out_text["prediction"].detach().cpu())
            img_feats.append(out_img["prediction"].detach().cpu())

        text_mat = torch.stack(text_feats, dim=0)
        img_mat = torch.stack(img_feats, dim=0)
        text_norm = torch.nn.functional.normalize(text_mat, dim=1)
        img_norm = torch.nn.functional.normalize(img_mat, dim=1)
        sim = text_norm @ img_norm.T
        pred_idx = sim.argmax(dim=1)
        hits = (pred_idx == torch.arange(n)).float().mean()
        hit_rate = float(hits)

        writer.add_scalar("task/s1_hit_rate", hit_rate, epoch)
        log_memory_read(logger, f"词↔图 检索命中率={hit_rate:.3f}")

    elif task == "sentences_images":
        path = data_root_path / "sentences_images.npz"
        if not path.exists():
            logger.warning(f"sentences_images toy data not found at {path}")
            return
        npz = np.load(path, allow_pickle=True)
        sentences = npz["sentences"]
        images = npz["images"]
        n = len(sentences)

        text_feats = []
        img_feats = []
        for i in range(n):
            # 句子直接作为一个 token 处理
            tokens = [str(sentences[i])]
            text_vec = encode_text(tokens, dim=128)
            img_tensor = torch.from_numpy(images[i])
            img_vec = encode_image(img_tensor, dim=128)

            drive_text = text_vec.unsqueeze(0).repeat(steps_eval, 1).to(device)
            drive_img = img_vec.unsqueeze(0).repeat(steps_eval, 1).to(device)

            out_text = model({"drive": drive_text}, steps=steps_eval)
            out_img = model({"drive": drive_img}, steps=steps_eval)
            text_feats.append(out_text["prediction"].detach().cpu())
            img_feats.append(out_img["prediction"].detach().cpu())

        text_mat = torch.stack(text_feats, dim=0)
        img_mat = torch.stack(img_feats, dim=0)
        text_norm = torch.nn.functional.normalize(text_mat, dim=1)
        img_norm = torch.nn.functional.normalize(img_mat, dim=1)
        sim = text_norm @ img_norm.T
        pred_idx = sim.argmax(dim=1)
        hits = (pred_idx == torch.arange(n)).float().mean()
        hit_rate = float(hits)

        writer.add_scalar("task/s2_hit_rate", hit_rate, epoch)
        log_memory_read(logger, f"句↔图 检索命中率={hit_rate:.3f}")

    elif task == "video_events":
        path = data_root_path / "video_events.npz"
        if not path.exists():
            logger.warning(f"video_events toy data not found at {path}")
            return
        npz = np.load(path, allow_pickle=True)
        videos = npz["videos"]  # [N, T, 1, 16, 16]
        labels = npz["labels"]
        n = len(labels)

        vid_feats = []
        for i in range(n):
            frames = torch.from_numpy(videos[i])
            vid_vec = encode_video(frames, dim=128)
            drive_vid = vid_vec.unsqueeze(0).repeat(steps_eval, 1).to(device)
            out_vid = model({"drive": drive_vid}, steps=steps_eval)
            vid_feats.append(out_vid["prediction"].detach().cpu())

        vid_mat = torch.stack(vid_feats, dim=0)
        vid_norm = torch.nn.functional.normalize(vid_mat, dim=1)
        sim = vid_norm @ vid_norm.T
        pred_idx = sim.argmax(dim=1)
        hits = (pred_idx == torch.arange(n)).float().mean()
        hit_rate = float(hits)

        writer.add_scalar("task/s3_self_retrieval", hit_rate, epoch)
        log_memory_read(logger, f"视频事件自检索命中率={hit_rate:.3f}")



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
    writer = SummaryWriter(logdir)

    # 优先使用命令行 duration，其次 runtime.duration_s
    duration = args.duration or int(cfg.get("runtime", {}).get("duration_s", 180))

    task = cfg.get("task", "unspecified")
    logger.info(f"启动训练 task={task}, duration={duration}")

    try:
        for epoch in range(3):
            out = model(steps=duration)

            # 文本日志
            log_pacemaker(logger, "θ/γ 节律=..., 噪声σ=... 已启动")
            log_self(logger, model.self_model.report())

            # 基本 loss
            loss = out["prediction_error"].abs().mean()
            writer.add_scalar("loss/train", float(loss.detach()), epoch)

            # 奖励（这里用 confidence 近似内在奖励，占位符 0 表示外在奖励）
            meta = out["meta"]
            r_int = float(meta["confidence"].detach())
            r_ext = 0.0
            writer.add_scalar("reward/r_int", r_int, epoch)
            writer.add_scalar("reward/r_ext", r_ext, epoch)
            # 与 README 中命名保持一致的标量
            writer.add_scalar("r_int", r_int, epoch)
            writer.add_scalar("r_ext", r_ext, epoch)

            # 点火 & 分支系数
            ignition_rate = float(out["ignition_rate"])
            branching_kappa = float(out["branching_kappa"])
            writer.add_scalar("mcc/ignition_rate", ignition_rate, epoch)
            writer.add_scalar("mcc/branching_kappa", branching_kappa, epoch)
            writer.add_scalar("ignition_rate", ignition_rate, epoch)
            writer.add_scalar("branching_kappa", branching_kappa, epoch)

            # 发放率与 synops 估计（能耗曲线）
            spikes = out["spikes"].float()
            spikes_per_s = float(spikes.mean() * 1000.0 / max(model_cfg.dt_ms, 1e-3))
            synops_est = float(spikes.numel())
            writer.add_scalar("energy/spikes_per_s", spikes_per_s, epoch)
            writer.add_scalar("energy/synops", synops_est, epoch)
            writer.add_scalar("spikes_per_s", spikes_per_s, epoch)
            writer.add_scalar("synops", synops_est, epoch)

            act_out = out["act_out"]
            epoch_energy = float(act_out.get("energy", 0.0))
            writer.add_scalar("energy/curve", epoch_energy, epoch)

            # Router 使用情况（Top-K 激活比例 & 概率分布）
            router_stats = out["router_stats"]
            probs = router_stats["probs"]
            topk_idx = router_stats["topk"]
            writer.add_scalar("router/moe_energy_ratio", float(out["moe_energy_ratio"]), epoch)
            # 总体 Top-K 使用率（与 README 中 router/topk_usage 对应）
            if probs.numel() > 0:
                topk_mean = float(probs[topk_idx].mean())
                writer.add_scalar("router/topk_usage", topk_mean, epoch)
            for i, p in enumerate(probs):
                writer.add_scalar(f"router/topk_usage/expert_{i}", float(p), epoch)

            # 意图效用（Intent Utility）
            utilities = out["utilities"]
            if utilities.numel() > 0:
                util_mean = float(utilities.mean().detach())
                util_max = float(utilities.max().detach())
                writer.add_scalar("agency/intent_utility_mean", util_mean, epoch)
                writer.add_scalar("agency/intent_utility_best", util_max, epoch)
                writer.add_histogram("agency/intent_utilities", utilities.detach(), epoch)

            # 承诺曲线（Commit Rate Curve）
            commit_state = out["commit_state"]
            committed = 1.0 if commit_state.get("committed", False) else 0.0
            writer.add_scalar("agency/commit_rate_curve", committed, epoch)

            # 自我信用分
            writer.add_scalar("self/credit", float(out["self_credit"]), epoch)

            # ---- 中文关键节点汇总日志（按模板） ----
            log_ignition(
                logger,
                f"窗口内跨模块同步提升，已广播；点火率={ignition_rate:.3f}，κ={branching_kappa:.3f}",
            )
            log_router(
                logger,
                f"Top-K 专家={topk_idx.tolist()}；均衡损失={float(router_stats['balance_loss']):.4e}",
            )
            mae = float(out["prediction_error"].abs().mean().detach())
            log_self_think(logger, f"目标=候选集；预期LP≈{-mae:.3f} 赋能=占位")
            if commit_state.get("committed", False):
                log_self_want(
                    logger,
                    f"承诺=True；预算≈{epoch_energy:.3f}",
                )
            else:
                log_self_want(logger, "承诺=False；预算=保守")
            log_self_do(
                logger,
                f"规划=单步动作；efference={float(act_out.get('efference', 0.0)):.3f}",
            )
            moe_ratio = float(out["moe_energy_ratio"])
            saving_pct = max(0.0, 1.0 - moe_ratio) * 100.0
            log_energy(
                logger,
                f"epoch synops={synops_est:.0f}（相对密集↓ {saving_pct:.1f}%），spikes/s={spikes_per_s:.2f}",
            )

            # 栅格图像（截取前 64x64，灰度图）
            t_max = min(spikes.shape[0], 64)
            n_max = min(spikes.shape[1], 64)
            raster = spikes[:t_max, :n_max]
            img = raster.unsqueeze(0)  # [1, T, N] 作为单通道图像
            writer.add_image("figs/raster_spikes", img, epoch, dataformats="CHW")

            # 反向传播与简单 SGD
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= 1e-4 * p.grad
                    p.grad.zero_()
            # 任务级指标（S1/S2/S3）
            if task in ("words_images", "sentences_images", "video_events"):
                compute_task_metrics(
                    model=model,
                    task=task,
                    writer=writer,
                    logger=logger,
                    epoch=epoch,
                    steps_eval=min(30, duration),
                    data_root="data/toy",
                )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
