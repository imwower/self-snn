import json
import subprocess
import sys
from pathlib import Path

from self_snn.api.infer_brain import BrainInput, run_brain_step


def test_run_brain_step_snapshot_fields():
    snapshot = run_brain_step("configs/agency.yaml", BrainInput(task_id="dummy"), steps=5)
    assert snapshot.region_activity
    assert snapshot.global_metrics
    assert snapshot.memory_summary
    assert snapshot.decision_hint
    assert "mode" in snapshot.decision_hint


def test_run_brain_infer_cli(tmp_path):
    script = Path("scripts/run_brain_infer.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            "configs/agency.yaml",
            "--task-id",
            "dummy",
            "--features",
            "{}",
            "--steps",
            "5",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    out = proc.stdout.strip()
    assert out
    data = json.loads(out)
    for key in ["region_activity", "global_metrics", "memory_summary", "decision_hint"]:
        assert key in data
