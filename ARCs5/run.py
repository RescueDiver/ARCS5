from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "data.json"
OUTPUT_FOLDER = PROJECT_ROOT / "task_groups"

GROUP_NAMES = [
    "small",
    "small_medium",
    "medium",
    "medium_large",
    "large",
    "very_large",
    "wtf",
]


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass(frozen=True)
class TaskSize:
    task_id: str

    largest_height: int
    largest_width: int
    largest_cells: int

    smallest_height: int
    smallest_width: int
    smallest_cells: int

    train_input_count: int


# =============================================================================
# LOADING
# =============================================================================

def load_data(data_file: Path) -> dict[str, Any]:
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find data file:\n{data_file}"
        )

    with data_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected data.json to contain a dictionary of task IDs."
        )

    return data


# =============================================================================
# GRID MEASUREMENT
# =============================================================================

def measure_grid(grid: list[list[int]]) -> tuple[int, int, int]:
    if not isinstance(grid, list) or not grid:
        return 0, 0, 0

    height = len(grid)

    widths = [
        len(row)
        for row in grid
        if isinstance(row, list)
    ]

    width = max(widths, default=0)
    cells = height * width

    return height, width, cells


def measure_task(task_id: str, task: dict[str, Any]) -> TaskSize:
    train_pairs = task.get("train", [])

    measurements: list[tuple[int, int, int]] = []

    for pair in train_pairs:
        input_grid = pair.get("input", [])
        measurements.append(measure_grid(input_grid))

    valid_measurements = [
        measurement
        for measurement in measurements
        if measurement[2] > 0
    ]

    if not valid_measurements:
        return TaskSize(
            task_id=task_id,
            largest_height=0,
            largest_width=0,
            largest_cells=0,
            smallest_height=0,
            smallest_width=0,
            smallest_cells=0,
            train_input_count=len(train_pairs),
        )

    largest = max(
        valid_measurements,
        key=lambda item: (item[2], item[0], item[1]),
    )

    smallest = min(
        valid_measurements,
        key=lambda item: (item[2], item[0], item[1]),
    )

    return TaskSize(
        task_id=task_id,
        largest_height=largest[0],
        largest_width=largest[1],
        largest_cells=largest[2],
        smallest_height=smallest[0],
        smallest_width=smallest[1],
        smallest_cells=smallest[2],
        train_input_count=len(train_pairs),
    )


def measure_all_tasks(data: dict[str, Any]) -> list[TaskSize]:
    measurements = [
        measure_task(task_id, task)
        for task_id, task in data.items()
    ]

    measurements.sort(
        key=lambda task: (
            task.largest_cells,
            task.largest_height,
            task.largest_width,
            task.task_id,
        )
    )

    return measurements


# =============================================================================
# NATURAL SIZE BREAKS
# =============================================================================

def find_natural_breaks(
    measurements: list[TaskSize],
    group_count: int,
) -> list[int]:
    """
    Finds natural breaks by locating the largest gaps between neighboring
    task sizes.

    Example:

        100 cells
        110 cells
        121 cells
        300 cells  <- large gap creates a group boundary

    The returned values are indexes where a new group begins.
    """

    task_count = len(measurements)

    if task_count <= 1:
        return []

    desired_break_count = min(
        group_count - 1,
        task_count - 1,
    )

    gap_candidates: list[tuple[int, int]] = []

    for index in range(task_count - 1):
        current_cells = measurements[index].largest_cells
        next_cells = measurements[index + 1].largest_cells

        gap = next_cells - current_cells

        if gap > 0:
            new_group_index = index + 1
            gap_candidates.append((gap, new_group_index))

    # Largest gaps first.
    gap_candidates.sort(
        key=lambda item: (-item[0], item[1])
    )

    chosen_breaks = [
        break_index
        for _, break_index in gap_candidates[:desired_break_count]
    ]

    # If there are not enough different size values, fill the remaining
    # boundaries using evenly distributed task positions.
    if len(chosen_breaks) < desired_break_count:
        for group_number in range(1, group_count):
            suggested_index = round(
                task_count * group_number / group_count
            )

            suggested_index = max(
                1,
                min(suggested_index, task_count - 1),
            )

            if suggested_index not in chosen_breaks:
                chosen_breaks.append(suggested_index)

            if len(chosen_breaks) >= desired_break_count:
                break

    chosen_breaks = sorted(
        set(chosen_breaks)
    )

    return chosen_breaks[:desired_break_count]


def split_into_groups(
    measurements: list[TaskSize],
    group_names: list[str],
) -> dict[str, list[TaskSize]]:
    breaks = find_natural_breaks(
        measurements=measurements,
        group_count=len(group_names),
    )

    groups: dict[str, list[TaskSize]] = {
        group_name: []
        for group_name in group_names
    }

    boundaries = [0, *breaks, len(measurements)]

    for group_index, group_name in enumerate(group_names):
        start = boundaries[group_index]
        end = boundaries[group_index + 1]

        groups[group_name] = measurements[start:end]

    return groups


# =============================================================================
# REPORTING
# =============================================================================

def format_task_line(
    task: TaskSize,
    number: int | None = None,
) -> str:
    number_text = ""

    if number is not None:
        number_text = f"{number:>3}. "

    shape = (
        f"{task.largest_height}x"
        f"{task.largest_width}"
    )

    return (
        f"{number_text}"
        f"{task.task_id:<12} "
        f"{shape:>9} "
        f"{task.largest_cells:>5} cells"
    )


def build_size_distribution_report(
    measurements: list[TaskSize],
) -> list[str]:
    lines = [
        "TASK INPUT SIZES",
        "=" * 72,
        "",
    ]

    for number, task in enumerate(measurements, start=1):
        lines.append(
            format_task_line(task, number)
        )

    lines.append("")

    return lines


def build_group_report(
    groups: dict[str, list[TaskSize]],
) -> list[str]:
    lines = [
        "ARCs5 FIRST-LOOK SIZE GROUPS",
        "=" * 72,
        "",
    ]

    total_tasks = sum(
        len(tasks)
        for tasks in groups.values()
    )

    lines.append(f"TOTAL TASKS: {total_tasks}")
    lines.append("")

    for group_name, tasks in groups.items():
        title = group_name.upper().replace("_", " ")

        lines.extend(
            [
                title,
                "-" * 72,
                f"Task count: {len(tasks)}",
            ]
        )

        if tasks:
            minimum = tasks[0].largest_cells
            maximum = tasks[-1].largest_cells

            lines.append(
                f"Cell range: {minimum} to {maximum}"
            )
            lines.append("")

            for number, task in enumerate(tasks, start=1):
                lines.append(
                    format_task_line(task, number)
                )
        else:
            lines.append("")
            lines.append("No tasks were assigned.")

        lines.extend(["", ""])

    return lines


def print_live_assignments(
    measurements: list[TaskSize],
    groups: dict[str, list[TaskSize]],
) -> None:
    task_to_group: dict[str, str] = {}

    for group_name, tasks in groups.items():
        for task in tasks:
            task_to_group[task.task_id] = group_name

    total = len(measurements)

    print()
    print("ARCs5 SIZE GROUPING")
    print("=" * 72)

    for number, task in enumerate(measurements, start=1):
        group_name = task_to_group.get(
            task.task_id,
            "unknown",
        )

        shape = (
            f"{task.largest_height}x"
            f"{task.largest_width}"
        )

        print(
            f"[{number:03}/{total:03}] "
            f"{task.task_id} | "
            f"largest input {shape:<7} | "
            f"{task.largest_cells:>4} cells | "
            f"{group_name.upper()}"
        )


# =============================================================================
# SAVING
# =============================================================================

def prepare_output_folder(output_folder: Path) -> None:
    if output_folder.exists():
        shutil.rmtree(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_json(
    file_path: Path,
    data: Any,
) -> None:
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def save_text(
    file_path: Path,
    lines: list[str],
) -> None:
    file_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def save_group_files(
    data: dict[str, Any],
    groups: dict[str, list[TaskSize]],
    output_folder: Path,
) -> None:
    for group_name, tasks in groups.items():
        group_data = {
            task.task_id: data[task.task_id]
            for task in tasks
        }

        save_json(
            output_folder / f"{group_name}.json",
            group_data,
        )


def save_inventory(
    groups: dict[str, list[TaskSize]],
    output_folder: Path,
) -> None:
    inventory: dict[str, Any] = {
        "grouping_method": (
            "Largest training input grid, grouped using "
            "the largest natural gaps in total cell count."
        ),
        "group_order": GROUP_NAMES,
        "groups": {},
        "tasks": {},
    }

    for group_name, tasks in groups.items():
        if tasks:
            minimum_cells = min(
                task.largest_cells
                for task in tasks
            )
            maximum_cells = max(
                task.largest_cells
                for task in tasks
            )
        else:
            minimum_cells = None
            maximum_cells = None

        inventory["groups"][group_name] = {
            "task_count": len(tasks),
            "minimum_cells": minimum_cells,
            "maximum_cells": maximum_cells,
            "task_ids": [
                task.task_id
                for task in tasks
            ],
        }

        for task in tasks:
            inventory["tasks"][task.task_id] = {
                "size_group": group_name,
                "largest_train_input": {
                    "height": task.largest_height,
                    "width": task.largest_width,
                    "cells": task.largest_cells,
                },
                "smallest_train_input": {
                    "height": task.smallest_height,
                    "width": task.smallest_width,
                    "cells": task.smallest_cells,
                },
                "train_input_count": task.train_input_count,
            }

    save_json(
        output_folder / "task_inventory.json",
        inventory,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    data = load_data(DATA_FILE)

    measurements = measure_all_tasks(data)

    if not measurements:
        print("No tasks were found in data.json.")
        return

    groups = split_into_groups(
        measurements=measurements,
        group_names=GROUP_NAMES,
    )

    prepare_output_folder(OUTPUT_FOLDER)

    save_group_files(
        data=data,
        groups=groups,
        output_folder=OUTPUT_FOLDER,
    )

    save_inventory(
        groups=groups,
        output_folder=OUTPUT_FOLDER,
    )

    size_report = build_size_distribution_report(
        measurements
    )

    group_report = build_group_report(groups)

    save_text(
        OUTPUT_FOLDER / "size_distribution.txt",
        size_report,
    )

    save_text(
        OUTPUT_FOLDER / "group_report.txt",
        group_report,
    )

    print_live_assignments(
        measurements=measurements,
        groups=groups,
    )

    print()
    print("\n".join(group_report))

    print("FILES CREATED")
    print("-" * 72)

    for group_name in GROUP_NAMES:
        group_file = OUTPUT_FOLDER / f"{group_name}.json"
        task_count = len(groups[group_name])

        print(
            f"{group_file.name:<24} "
            f"{task_count:>3} tasks"
        )

    print()
    print("Additional reports:")
    print("  task_inventory.json")
    print("  size_distribution.txt")
    print("  group_report.txt")
    print()
    print(f"Output folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()