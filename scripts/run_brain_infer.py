import argparse
import json

from self_snn.api.infer_brain import BrainInput, run_brain_step, snapshot_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 self-snn 在线脑推理，输出 BrainSnapshot JSON")
    parser.add_argument("--config", type=str, default="configs/agency.yaml", help="配置文件路径")
    parser.add_argument("--task-id", type=str, required=True, help="任务 ID 或场景名称")
    parser.add_argument("--text", type=str, default="", help="任务文本描述")
    parser.add_argument("--features", type=str, default="{}", help="JSON 字符串形式的任务特征")
    parser.add_argument("--steps", type=int, default=10, help="前向步数")
    parser.add_argument("--device", type=str, default=None, help="运行设备")
    args = parser.parse_args()

    try:
        features = json.loads(args.features) if args.features else {}
    except json.JSONDecodeError:
        features = {}

    brain_input = BrainInput(task_id=args.task_id, text_summary=args.text, features=features)
    snapshot = run_brain_step(args.config, brain_input, steps=int(args.steps), device=args.device)
    data = snapshot_to_dict(snapshot)
    print(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()
