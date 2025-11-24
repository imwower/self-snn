import json
import subprocess
import sys
from pathlib import Path


def test_dump_brain_graph_json(tmp_path):
    script = Path("scripts/dump_brain_graph.py")
    config = Path("configs/agency.yaml")
    output_path = tmp_path / "brain_graph.json"

    subprocess.run(
        [sys.executable, str(script), "--config", str(config), "--output", str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.is_file()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)

    region_ids = {r.get("id") for r in data.get("regions", [])}
    assert {"mcc", "router", "agency"}.issubset(region_ids)

    metric_names = {m.get("name") for m in data.get("metrics", [])}
    assert {"num_regions", "router_M", "delay_dmax"}.issubset(metric_names)

    assert data.get("meta", {}).get("version") == "r7_brain_graph_v1"
