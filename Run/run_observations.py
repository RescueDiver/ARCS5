import json
from collections import defaultdict
from pathlib import Path
from typing import Any


# =============================================================================
# WHAT GROUP SHOULD BE SPLIT?
# =============================================================================

# Examples:
# INPUT_GROUP = "small_medium.json"
# INPUT_GROUP = "medium.json"
# INPUT_GROUP = "small_medium/same_size.json"
#
# The path is relative to ARCs5/task_groups.
INPUT_GROUP = "small_medium.json"


# =============================================================================
# PATHS
# =============================================================================

# This file is expected to be located here:
#
# ARCs5/
#     Run/
#         run_observation.py
#
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUP_ROOT = PROJECT_ROOT / "task_groups"

INPUT_FILE = GROUP_ROOT / INPUT_GROUP


# =============================================================================
# BASIC GRID MEASUREMENTS
# =============================================================================

def grid_shape(grid: Any) -> tuple[int, int]:
    """
    Return the grid height and width.

    Empty or invalid grids return (0, 0).
    """
    if not isinstance(grid, list) or not grid:
        return 0, 0

    valid_rows = [
        row
        for row in grid
        if isinstance(row, list)
    ]

    if not valid_rows:
        return 0, 0

    height = len(valid_rows)
    width = max(
        (len(row) for row in valid_rows),
        default=0,
    )

    return height, width


# =============================================================================
# ONE OBSERVATION:
# INPUT SIZE RELATIONSHIP TO OUTPUT SIZE
# =============================================================================

def observe_pair_size_relationship(
    input_grid: Any,
    output_grid: Any,
) -> str:
    """
    Observe only the relationship between input and output dimensions.

    No task rule is guessed here.
    """
    input_height, input_width = grid_shape(input_grid)
    output_height, output_width = grid_shape(output_grid)

    if input_height == 0 or input_width == 0:
        return "invalid_grid"

    if output_height == 0 or output_width == 0:
        return "invalid_grid"

    if (
        input_height == output_height
        and input_width == output_width
    ):
        return "same_size"

    height_change = output_height - input_height
    width_change = output_width - input_width

    if height_change > 0 and width_change > 0:
        return "both_dimensions_grow"

    if height_change < 0 and width_change < 0:
        return "both_dimensions_shrink"

    if height_change == 0 and width_change > 0:
        return "width_grows"

    if height_change == 0 and width_change < 0:
        return "width_shrinks"

    if height_change > 0 and width_change == 0:
        return "height_grows"

    if height_change < 0 and width_change == 0:
        return "height_shrinks"

    if height_change > 0 and width_change < 0:
        return "taller_and_narrower"

    if height_change < 0 and width_change > 0:
        return "shorter_and_wider"

    return "other_size_change"


def observe_task(task: dict[str, Any]) -> tuple[str, list[str]]:
    """
    Observe all training pairs.

    If every training pair has the same size relationship, the task goes into
    that group.

    If its training pairs have different relationships, it goes into a
    mixed group whose name records the observed relationships.
    """
    train_pairs = task.get("train", [])

    if not isinstance(train_pairs, list) or not train_pairs:
        return "no_train_pairs", []

    pair_relationships: list[str] = []

    for pair in train_pairs:
        if not isinstance(pair, dict):
            pair_relationships.append("invalid_grid")
            continue

        relationship = observe_pair_size_relationship(
            pair.get("input"),
            pair.get("output"),
        )

        pair_relationships.append(relationship)

    unique_relationships = sorted(set(pair_relationships))

    if len(unique_relationships) == 1:
        return unique_relationships[0], pair_relationships

    mixed_name = "mixed__" + "__".join(unique_relationships)

    return mixed_name, pair_relationships


# =============================================================================
# FILE LOADING AND SAVING
# =============================================================================

def load_tasks(file_path: Path) -> dict[str, dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(
            "The selected group file was not found:\n"
            f"{file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"{file_path.name} must contain a dictionary of ARC tasks."
        )

    tasks: dict[str, dict[str, Any]] = {}

    for task_id, task in data.items():
        if isinstance(task, dict):
            tasks[str(task_id)] = task

    return tasks


def output_folder_for(input_file: Path) -> Path:
    """
    Example:

    task_groups/small_medium.json
        becomes
    task_groups/small_medium/output_size_relationship/

    A deeper file such as:

    task_groups/small_medium/same_size.json
        becomes
    task_groups/small_medium/same_size/output_size_relationship/
    """
    return (
        input_file.parent
        / input_file.stem
        / "output_size_relationship"
    )


def save_json(
    file_path: Path,
    data: Any,
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
        )


# =============================================================================
# REPORTING
# =============================================================================

def safe_file_name(group_name: str) -> str:
    cleaned = "".join(
        character
        if character.isalnum() or character in {"_", "-"}
        else "_"
        for character in group_name
    )

    return cleaned.strip("_") or "unnamed_group"


def print_task_observation(
    task_number: int,
    task_count: int,
    task_id: str,
    group_name: str,
    pair_relationships: list[str],
) -> None:
    pair_text = ", ".join(pair_relationships)

    print(
        f"[{task_number:03d}/{task_count:03d}] "
        f"{task_id} | "
        f"{pair_text:<45} | "
        f"{group_name.upper()}"
    )


def write_report(
    output_folder: Path,
    input_file: Path,
    groups: dict[str, dict[str, Any]],
    task_observations: dict[str, Any],
) -> None:
    lines: list[str] = []

    lines.append("ARCs5 OUTPUT SIZE RELATIONSHIP GROUPING")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Input group : {input_file}")
    lines.append(
        f"Total tasks : "
        f"{sum(len(tasks) for tasks in groups.values())}"
    )
    lines.append(f"New groups : {len(groups)}")
    lines.append("")

    for group_name in sorted(groups):
        task_ids = sorted(groups[group_name])

        lines.append(group_name.upper())
        lines.append("-" * 72)
        lines.append(f"Task count: {len(task_ids)}")

        for number, task_id in enumerate(task_ids, start=1):
            relationships = task_observations[task_id][
                "train_pair_relationships"
            ]

            relationship_text = ", ".join(relationships)

            lines.append(
                f"{number:3d}. "
                f"{task_id:<12} "
                f"{relationship_text}"
            )

        lines.append("")

    report_file = output_folder / "group_report.txt"

    report_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print()
    print("ARCs5 OUTPUT SIZE RELATIONSHIP GROUPING")
    print("=" * 72)
    print(f"Input group: {INPUT_FILE}")
    print()

    tasks = load_tasks(INPUT_FILE)

    grouped_tasks: dict[str, dict[str, Any]] = defaultdict(dict)
    task_observations: dict[str, Any] = {}

    task_count = len(tasks)

    for task_number, (task_id, task) in enumerate(
        tasks.items(),
        start=1,
    ):
        group_name, pair_relationships = observe_task(task)

        grouped_tasks[group_name][task_id] = task

        task_observations[task_id] = {
            "source_group": INPUT_GROUP,
            "result_group": group_name,
            "train_pair_relationships": pair_relationships,
        }

        print_task_observation(
            task_number=task_number,
            task_count=task_count,
            task_id=task_id,
            group_name=group_name,
            pair_relationships=pair_relationships,
        )

    output_folder = output_folder_for(INPUT_FILE)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old_json_file in output_folder.glob("*.json"):
        old_json_file.unlink()

    for group_name, group_tasks in sorted(grouped_tasks.items()):
        file_name = safe_file_name(group_name) + ".json"

        save_json(
            output_folder / file_name,
            group_tasks,
        )

    save_json(
        output_folder / "task_observations.json",
        task_observations,
    )

    write_report(
        output_folder=output_folder,
        input_file=INPUT_FILE,
        groups=grouped_tasks,
        task_observations=task_observations,
    )

    print()
    print("NEW GROUPS")
    print("-" * 72)

    for group_name, group_tasks in sorted(grouped_tasks.items()):
        print(
            f"{group_name:<48} "
            f"{len(group_tasks):>3} tasks"
        )

    print()
    print("FILES CREATED")
    print("-" * 72)

    for group_name, group_tasks in sorted(grouped_tasks.items()):
        file_name = safe_file_name(group_name) + ".json"

        print(
            f"{file_name:<52} "
            f"{len(group_tasks):>3} tasks"
        )

    print()
    print("Additional files:")
    print("  task_observations.json")
    print("  group_report.txt")
    print()
    print(f"Output folder: {output_folder}")
    print()


if __name__ == "__main__":
    main()

