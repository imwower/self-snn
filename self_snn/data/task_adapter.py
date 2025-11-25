from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch

from self_snn.utils.encoders import encode_text, encode_image, encode_video


def _encode_image_safe(path: str, dim: int = 128) -> np.ndarray:
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:
        return np.zeros(dim, dtype=np.float32)
    try:
        img = imageio.imread(path)
        vec = encode_image(torch.from_numpy(np.array(img)), dim=dim)
        return vec.detach().cpu().numpy().astype(np.float32)
    except Exception:
        return np.zeros(dim, dtype=np.float32)


def convert_generated_tasks_to_dataset(
    tasks: List[Dict[str, Any]],
    output_dir: str,
    kind: str,
) -> Dict[str, Any]:
    """
    将 me-agent 导出的 GeneratedTask 列表转换为 self-snn 训练可用的占位数据集。
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    text_features: List[np.ndarray] = []
    image_features: List[np.ndarray] = []
    labels: List[int] = []

    for idx, t in enumerate(tasks):
        payload = t.get("payload") if isinstance(t, dict) else {}
        question = ""
        if isinstance(payload, dict):
            question = str(payload.get("question") or payload.get("text") or "")
        if not question:
            question = str(t.get("id", f"task_{idx}"))
        text_vec = encode_text([question], dim=128)[0].detach().cpu().numpy().astype(np.float32)
        text_features.append(text_vec)

        img_path = ""
        if isinstance(payload, dict):
            img_path = payload.get("image_path") or payload.get("image") or ""
        if img_path and os.path.exists(img_path):
            img_vec = _encode_image_safe(str(img_path), dim=128)
        else:
            img_vec = np.zeros(128, dtype=np.float32)
        image_features.append(img_vec)

        labels.append(idx % 2)

    data = {
        "text": np.stack(text_features) if text_features else np.zeros((0, 128), dtype=np.float32),
        "image": np.stack(image_features) if image_features else np.zeros((0, 128), dtype=np.float32),
        "label": np.array(labels, dtype=np.int64),
        "kind": kind,
        "count": len(labels),
    }
    out_path = Path(output_dir) / "generated_tasks.npz"
    np.savez(out_path, **data)
    return {"dataset_path": str(out_path), "count": len(labels), "kind": kind}


__all__ = ["convert_generated_tasks_to_dataset"]
