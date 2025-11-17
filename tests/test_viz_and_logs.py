from pathlib import Path

import torch

from self_snn.utils.viz import (
    save_raster,
    save_curve,
    save_heatmap,
    save_intent_utility_with_components,
)
from self_snn.utils.logging_cn import (
    setup_logger,
    log_pacemaker,
    log_ignition,
    log_memory_write,
    log_memory_read,
    log_router,
    log_self,
    log_self_think,
    log_self_want,
    log_self_do,
    log_energy,
    log_consistency,
)


def test_viz_outputs_fig_files(tmp_path):
    """可视化工具应能生成指定文件名的图像。"""
    figs_dir = tmp_path / "figs"

    # 栅格图
    spikes = (torch.rand(10, 5) > 0.7).to(torch.float32)
    save_raster(spikes, figs_dir / "raster_spikes.png")

    # 曲线图（如 vm_trace, energy_curve, agency_timeline, self_credit）
    values = [0.1 * i for i in range(10)]
    save_curve(values, figs_dir / "vm_trace.png")
    save_curve(values, figs_dir / "energy_curve.png")
    save_curve(values, figs_dir / "agency_timeline.png")
    save_curve(values, figs_dir / "self_credit.png")
    save_curve(values, figs_dir / "commit_rate_curve.png")

    # 热力图（如 ignition_heatmap, router_usage）
    mat = torch.randn(8, 4)
    save_heatmap(mat, figs_dir / "ignition_heatmap.png")
    save_heatmap(mat, figs_dir / "router_usage.png")

    # 意图效用 + 分项条形图
    best_utils = [0.2 * i for i in range(5)]
    components = {
        "LP_hat": 0.5,
        "Empower_hat": 0.3,
        "R_ext_hat": 0.2,
        "Energy": 0.1,
        "Risk": 0.05,
        "Boredom": 0.02,
        "U": 0.4,
    }
    save_intent_utility_with_components(best_utils, components, figs_dir / "intent_utility.png")

    # delay_replay_error 在 eval_memory 中生成，这里只验证可用 curve 函数
    save_curve(values, figs_dir / "delay_replay_error.png")

    # 所有关键文件应存在
    expected = [
        "raster_spikes.png",
        "vm_trace.png",
        "ignition_heatmap.png",
        "delay_replay_error.png",
        "router_usage.png",
        "energy_curve.png",
        "agency_timeline.png",
        "intent_utility.png",
        "commit_rate_curve.png",
        "self_credit.png",
    ]
    for name in expected:
        assert (figs_dir / name).is_file()


def test_logging_cn_prefixes(tmp_path):
    """中文日志前缀应按模板输出到 train.log。"""
    logger = setup_logger(tmp_path, name=f"self_snn_{tmp_path.name}")
    log_pacemaker(logger, "θ/γ 节律=..., 噪声σ=...")
    log_ignition(logger, "点火率=..., κ=...")
    log_memory_write(logger, "键相位=..., 序列长度=...")
    log_memory_read(logger, "命中率=..., 偏差=...")
    log_router(logger, "Top-K=..., z_loss=...")
    log_self(logger, "置信=..., 能耗=..., 风险=...")
    log_self_think(logger, "目标=..., 预期LP=...")
    log_self_want(logger, "承诺=..., 预算=...")
    log_self_do(logger, "规划=..., efference=...")
    log_energy(logger, "synops=..., ratio=...")
    log_consistency(logger, "信用分=..., Brier=..., ECE=...")

    log_file = Path(tmp_path) / "train.log"
    # FileHandler 写入是缓冲的，这里主动 flush 一下
    for handler in logger.handlers:
        handler.flush()
    text = log_file.read_text(encoding="utf-8")

    for prefix in [
        "【起搏器】",
        "【点火】",
        "【记忆-写入】",
        "【记忆-检索】",
        "【路由】",
        "【我】",
        "【我想】",
        "【我要】",
        "【我做】",
        "【节能】",
        "【自洽】",
    ]:
        assert prefix in text
