import json
import tempfile
from pathlib import Path
import unittest

from scripts.train_from_schedule import run_from_schedule


class TrainFromScheduleStubTest(unittest.TestCase):
    def test_dry_run_materializes_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            schedule = {
                "id": "sched-1",
                "repo_id": "self-snn",
                "config_path": "",
                "output_dir": tmpdir,
                "max_epochs": 1,
                "tasks": [
                    {"id": "t1", "template_id": "tpl", "payload": {"x": 1}, "expected_behavior": "do", "labels": {}}
                ],
            }
            schedule_path = Path(tmpdir) / "schedule.json"
            schedule_path.write_text(json.dumps(schedule), encoding="utf-8")

            result = run_from_schedule(str(schedule_path), dry_run=True)
            dataset_path = Path(result["dataset_path"])
            self.assertTrue(dataset_path.exists(), "dataset path should be created during dry-run")
            content = dataset_path.read_text(encoding="utf-8").strip()
            self.assertIn("\"id\": \"t1\"", content)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
