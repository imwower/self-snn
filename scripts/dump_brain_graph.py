import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _get(cfg: Dict[str, Any], keys: List[str], default: Any) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


def _sum_neurons(neurons) -> int:
    if isinstance(neurons, (list, tuple)):
        try:
            return int(sum(int(x) for x in neurons))
        except Exception:
            return 0
    try:
        return int(neurons)
    except Exception:
        return 0


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_brain_graph(cfg: Dict[str, Any], config_path: str | Path) -> Dict[str, Any]:
    mcc_cfg = cfg.get("mcc", {})
    memory_cfg = cfg.get("memory", {})
    router_cfg = cfg.get("router", {})
    agency_cfg = cfg.get("agency", {})
    runtime_cfg = cfg.get("runtime", {})
    safety_cfg = cfg.get("safety", {})

    pmc_n = int(_get(mcc_cfg, ["pmc", "N"], 64))
    gw_hubs = int(_get(mcc_cfg, ["gw", "hubs"], 160))
    gw_k_select = int(_get(mcc_cfg, ["gw", "k_select"], 2))
    neurons_per_uCol = _get(mcc_cfg, ["wm", "neurons_per_uCol"], [80, 20])
    wm_neurons = _sum_neurons(neurons_per_uCol)
    wm_microcolumns = int(_get(mcc_cfg, ["wm", "microcolumns"], 3))
    wm_size = wm_microcolumns * wm_neurons
    mcc_size = pmc_n + gw_hubs + wm_neurons

    delay_cfg = memory_cfg.get("delay", {})
    delay_dmax = int(delay_cfg.get("dmax", 120))

    experts_cfg = router_cfg.get("experts", {})
    router_M = int(experts_cfg.get("M", 16))
    router_K = int(experts_cfg.get("K", 2))
    router_z = float(_get(router_cfg, ["balance", "z_loss"], 1e-3))
    router_mask = router_cfg.get("mask_type", "binary")

    self_key_dim = int(agency_cfg.get("self_key_dim", 64))
    prospect_cfg = agency_cfg.get("prospect", {})
    n_candidates = int(prospect_cfg.get("n_candidates", 5))
    horizon = int(prospect_cfg.get("horizon", 0))
    agency_size = self_key_dim + n_candidates + horizon

    regions = [
        {
            "id": "mcc",
            "name": "MCC",
            "kind": "core",
            "size": mcc_size,
            "meta": {
                "pmc_N": pmc_n,
                "gw_hubs": gw_hubs,
                "wm_neurons_per_uCol": neurons_per_uCol,
            },
        },
        {
            "id": "d_mem",
            "name": "Delay Memory",
            "kind": "memory",
            "size": delay_dmax,
            "meta": {"dmax": delay_dmax},
        },
        {
            "id": "wm",
            "name": "World Model",
            "kind": "worldmodel",
            "size": wm_size,
            "meta": {"microcolumns": wm_microcolumns, "neurons_per_uCol": neurons_per_uCol},
        },
        {
            "id": "router",
            "name": "GW-Router/MoE",
            "kind": "router",
            "size": router_M,
            "meta": {"experts_M": router_M, "experts_K": router_K, "z_loss": router_z, "mask_type": router_mask},
        },
        {
            "id": "agency",
            "name": "Agency(Self)",
            "kind": "agency",
            "size": agency_size,
            "meta": {"self_key_dim": self_key_dim, "n_candidates": n_candidates, "horizon": horizon},
        },
    ]

    connections = [
        {"id": "mcc->d_mem", "pre_region": "mcc", "post_region": "d_mem", "type": "plastic", "sparsity": 0.5},
        {"id": "d_mem->wm", "pre_region": "d_mem", "post_region": "wm", "type": "plastic", "sparsity": 0.5},
        {"id": "mcc->router", "pre_region": "mcc", "post_region": "router", "type": "routing", "sparsity": 0.3},
        {"id": "router->agency", "pre_region": "router", "post_region": "agency", "type": "conditional", "sparsity": 0.3},
        {"id": "wm->agency", "pre_region": "wm", "post_region": "agency", "type": "predictive", "sparsity": 0.4},
    ]

    metrics = [
        {"name": "num_regions", "value": len(regions), "unit": ""},
        {"name": "num_connections", "value": len(connections), "unit": ""},
        {"name": "router_M", "value": router_M, "unit": ""},
        {"name": "router_K", "value": router_K, "unit": ""},
        {"name": "delay_dmax", "value": delay_dmax, "unit": "steps"},
    ]

    spike_budget = safety_cfg.get("spike_budget_per_s")
    if spike_budget is not None:
        metrics.append({"name": "spike_budget_per_s", "value": float(spike_budget), "unit": "spikes"})

    runtime_duration = runtime_cfg.get("duration_s")
    if runtime_duration is not None:
        metrics.append({"name": "runtime_duration_s", "value": float(runtime_duration), "unit": "s"})

    dt_ms = runtime_cfg.get("dt_ms", _get(mcc_cfg, ["pmc", "dt_ms"], 1.0))
    if dt_ms is not None:
        metrics.append({"name": "dt_ms", "value": float(dt_ms), "unit": "ms"})

    meta = {
        "config_path": str(config_path),
        "version": "r7_brain_graph_v1",
        "router_mask_type": router_mask,
        "gw_k_select": gw_k_select,
        "energy_module": "self_snn.utils.energy",
    }

    return {
        "regions": regions,
        "connections": connections,
        "metrics": metrics,
        "meta": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="导出简化 BrainGraph JSON")
    parser.add_argument("--config", type=str, default="configs/agency.yaml", help="YAML 配置路径")
    parser.add_argument("--output", type=str, default="", help="输出 JSON 路径；不填则打印到 stdout")
    args = parser.parse_args()

    cfg = load_config(args.config)
    graph = build_brain_graph(cfg, args.config)
    text = json.dumps(graph, ensure_ascii=False, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
