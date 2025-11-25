import os
import tempfile
import unittest

from self_snn.data.task_adapter import convert_generated_tasks_to_dataset


class TaskAdapterTest(unittest.TestCase):
    def test_convert_generated_tasks_to_dataset(self) -> None:
        tasks = [
            {"id": "t1", "payload": {"question": "hello", "image_path": ""}},
            {"id": "t2", "payload": {"question": "world", "image_path": ""}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            info = convert_generated_tasks_to_dataset(tasks, tmpdir, kind="multimodal")
            self.assertTrue(os.path.exists(info["dataset_path"]))
            self.assertEqual(info["count"], 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
