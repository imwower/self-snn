import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_from_schedule import run_from_schedule


class TrainFromScheduleRealSmallTest(unittest.TestCase):
    def test_real_small_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            schedule = {
                "id": "sched-small",
                "repo_id": "self-snn",
                "config_path": "configs/s0_minimal.yaml",
                "output_dir": str(Path(tmpdir) / "out"),
                "max_epochs": 1,
                "tasks": [
                    {"id": "t1", "payload": {"question": "hello"}, "expected_behavior": "", "labels": {}},
                ],
            }
            path = Path(tmpdir) / "sched.json"
            path.write_text(json.dumps(schedule), encoding="utf-8")
            result = run_from_schedule(str(path), dry_run=False, duration=1)
            self.assertEqual(result["status"], "ok")
            self.assertIn("metrics", result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
