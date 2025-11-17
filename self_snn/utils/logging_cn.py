from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(logdir: str | Path, name: str = "self_snn") -> logging.Logger:
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(logdir / "train.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
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
