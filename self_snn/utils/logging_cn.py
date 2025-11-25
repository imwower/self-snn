from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    logdir: str | Path,
    name: str = "self_snn",
    to_stdout: bool = False,
) -> logging.Logger:
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    if not logger.handlers:
        fh = logging.FileHandler(logdir / "train.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    # 可选输出到 stdout，便于调试时直接在控制台看到关键节点日志
    if to_stdout and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def log_pacemaker(logger: logging.Logger, msg: str) -> None:
    logger.info(f"【起搏器】{msg}")


def log_ignition(logger: logging.Logger, msg: str) -> None:
    logger.info(f"【点火】{msg}")


def log_memory_write(logger: logging.Logger, msg: str) -> None:
    logger.info(f"【记忆-写入】{msg}")


def log_memory_read(logger: logging.Logger, msg: str) -> None:
    logger.info(f"【记忆-检索】{msg}")


def log_router(logger: logging.Logger, msg: str) -> None:
    logger.info(f"【路由】{msg}")


def log_self(logger: logging.Logger, msg: str) -> None:
    logger.info(f"【我】{msg}")


def log_self_think(logger: logging.Logger, msg: str) -> None:
    logger.info(f"【我想】{msg}")


def log_self_want(logger: logging.Logger, msg: str) -> None:
    logger.info(f"【我要】{msg}")


def log_self_do(logger: logging.Logger, msg: str) -> None:
    logger.info(f"【我做】{msg}")


def log_energy(logger: logging.Logger, msg: str) -> None:
    logger.info(f"【节能】{msg}")


def log_consistency(logger: logging.Logger, msg: str) -> None:
    logger.info(f"【自洽】{msg}")
