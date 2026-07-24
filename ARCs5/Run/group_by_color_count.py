import json
from collections import defaultdict
from pathlib import Path


# Group to split, relative to ARCs5/task_groups
INPUT_GROUP = (
    "small_medium/"
    "output_size_relationship/"
    "same_size.json"
)

ROOT = Path(__file__).resolve().parent.parent
GROUP_ROOT = ROOT / "task_groups"
INPUT_FILE = GROUP_ROOT / INPUT_GROUP


def task_color_count(task: dict) -> int:
    """Count unique colors across every training input in the task."""
    colors = set()

    for pair in task.get("train", []):
        for row in pair.get("input", []):
            colors.update(row)

    return len(colors)


def main() -> None:
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        tasks = json.load(file)

    groups = defaultdict(dict)

    for task_id, task in tasks.items():
        count = task_color_count(task)
        groups[f"colors_{count}"][task_id] = task

    output_folder = INPUT_FILE.parent / INPUT_FILE.stem / "color_count"
    output_folder.mkdir(parents=True, exist_ok=True)

    for old_file in output_folder.glob("*.json"):
        old_file.unlink()

    print()
    print(f"Input: {INPUT_FILE}")
    print("-" * 70)

    for group_name, group_tasks in sorted(groups.items()):
        output_file = output_folder / f"{group_name}.json"

        with output_file.open("w", encoding="utf-8") as file:
            json.dump(group_tasks, file, indent=2)

        print(f"{group_name:<20} {len(group_tasks):>3} tasks")

    print()
    print(f"Output: {output_folder}")
    print()


if __name__ == "__main__":
    main()
