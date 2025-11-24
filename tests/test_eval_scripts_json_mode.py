import json
import subprocess
import sys
from pathlib import Path


def _run_json(script: str, args: list[str]) -> dict:
    cmd = [sys.executable, str(Path("scripts") / script), *args, "--json"]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = proc.stdout.strip()
    assert stdout, f"{script} returned empty stdout in JSON mode"
    return json.loads(stdout)


def test_eval_scripts_json_mode(tmp_path):
    ignition_metrics = _run_json(
        "eval_ignition.py",
        ["--steps", "60", "--logdir", str(tmp_path / "ignition")],
    )
    assert "ignition_rate" in ignition_metrics
    assert "branching_kappa" in ignition_metrics

    memory_metrics = _run_json(
        "eval_memory.py",
        [
            "--logdir",
            str(tmp_path / "memory"),
            "--trials",
            "3",
            "--seq_len",
            "8",
            "--delay",
            "3",
        ],
    )
    assert "cue_hit_rate" in memory_metrics
    assert "replay_time_error_ms" in memory_metrics

    router_metrics = _run_json(
        "eval_router_energy.py",
        ["--steps", "30", "--logdir", str(tmp_path / "router")],
    )
    assert "synops_ratio" in router_metrics
    assert "synops_sparse" in router_metrics
    assert "avg_spikes_per_s" in router_metrics
