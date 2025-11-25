"""
根据 me-agent 导出的 TrainSchedule 启动 self-snn 训练。
默认以 dry-run 形式验证流水线，避免在单元测试中运行重型训练。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from self_snn.utils.config_loader import load_config, build_self_snn_config
from self_snn.utils.logging_cn import setup_logger, log_self
from self_snn.data.task_adapter import convert_generated_tasks_to_dataset

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型提示
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from self_snn.core.workspace import SelfSNN


def load_schedule(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def tasks_to_dataset(tasks: List[Dict[str, Any]], out_dir: str) -> str:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / "train_schedule_tasks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    return str(out_path)


def run_from_schedule(schedule_path: str, dry_run: bool = False, duration: int | None = None) -> Dict[str, Any]:
    schedule = load_schedule(schedule_path)
    cfg_path = schedule.get("config_path")
    output_dir = schedule.get("output_dir") or "runs/from_schedule"
    tasks = schedule.get("tasks") or []
    dataset_path = tasks_to_dataset(tasks, output_dir)

    cfg = load_config(cfg_path) if cfg_path else {}
    cfg.setdefault("data", {})["generated_tasks"] = dataset_path
    cfg.setdefault("logging", {}).setdefault("logdir", output_dir)

    summary = {
        "train_schedule_id": schedule.get("id"),
        "task_count": len(tasks),
        "dataset_path": dataset_path,
        "config_path": cfg_path,
        "logdir": output_dir,
    }

    if dry_run:
        print(json.dumps({"status": "dry_run", **summary}, ensure_ascii=False))
        return summary

    dataset_info = convert_generated_tasks_to_dataset(tasks, output_dir, kind=schedule.get("repo_id", "self-snn"))
    cfg["data"]["generated_tasks"] = dataset_info["dataset_path"]
    summary["dataset_path"] = dataset_info["dataset_path"]

    import torch  # 延迟导入，避免 dry-run 时强依赖 torch
    from torch.utils.tensorboard import SummaryWriter
    from self_snn.core.workspace import SelfSNN

    model_cfg = build_self_snn_config(cfg)
    device = torch.device(model_cfg.device)
    model = SelfSNN(model_cfg).to(device)
    logger = setup_logger(output_dir, to_stdout=True)
    writer = SummaryWriter(output_dir)

    run_duration = duration or int(cfg.get("runtime", {}).get("duration_s", 60))
    max_epochs = int(schedule.get("max_epochs", 1))

    for epoch in range(max_epochs):
        out = model(steps=run_duration)
        loss = out["prediction_error"].abs().mean()
        writer.add_scalar("loss/train", float(loss.detach()), epoch)
        writer.add_scalar("self/confidence", float(out["meta"]["confidence"]), epoch)
        log_self(logger, model.self_model.report())

    metrics = {"last_loss": float(loss.detach()), "epochs": max_epochs}
    result = {"status": "ok", **summary, "metrics": metrics}
    print(json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-schedule", required=True, help="TrainSchedule JSON 路径")
    parser.add_argument("--dry-run", action="store_true", help="仅验证输入/输出与数据准备，不实际训练")
    parser.add_argument("--duration", type=int, default=0, help="覆盖 schedule/config 中的 duration")
    args = parser.parse_args()
    run_from_schedule(args.train_schedule, dry_run=args.dry_run, duration=args.duration or None)


if __name__ == "__main__":  # pragma: no cover
    main()
