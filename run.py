from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "data.json"
OUTPUT_FOLDER = PROJECT_ROOT / "task_groups"


# =============================================================================
# GROUP NAMES
# =============================================================================

SIZE_GROUP_NAMES = [
    "small",
    "small_medium",
    "medium",
    "medium_large",
    "large",
    "very_large",
    "wtf",
]

SCENE_GROUP_NAMES = [
    "object",
    "pattern",
    "mixed",
    "uncertain",
]

BACKGROUND_GROUP_NAMES = [
    "clear",
    "possible",
    "none",
    "multiple",
    "uncertain",
]


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass(frozen=True)
class GridFacts:
    height: int
    width: int
    cells: int
    rectangular: bool

    colors: tuple[int, ...]
    color_count: int
    color_counts: dict[int, int]

    most_common_color: int | None
    most_common_count: int
    dominant_fraction: float

    border_counts: dict[int, int]
    most_common_border_color: int | None
    border_dominant_fraction: float

    full_horizontal_lines: tuple[int, ...]
    full_vertical_lines: tuple[int, ...]

    repeated_row_pairs: int
    repeated_column_pairs: int
    repeated_2x2_count: int

    four_connected_component_count: int
    eight_connected_component_count: int
    largest_four_component_fraction: float
    largest_eight_component_fraction: float


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


@dataclass
class TaskClassification:
    task_id: str
    size_group: str = "unknown"

    output_size_group: str = "unknown"
    train_pair_output_size_groups: list[str] | None = None

    scene_group: str = "uncertain"
    scene_scores: dict[str, float] | None = None

    background_group: str = "uncertain"
    background_scores: dict[str, float] | None = None
    background_candidates: list[int] | None = None

    structure_tags: list[str] | None = None
    measurement_tags: list[str] | None = None

    largest_train_input: dict[str, int] | None = None
    smallest_train_input: dict[str, int] | None = None
    train_input_count: int = 0

    train_input_facts: list[dict[str, Any]] | None = None
    train_output_facts: list[dict[str, Any]] | None = None


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
# GRID HELPERS
# =============================================================================

def is_grid(grid: Any) -> bool:
    if not isinstance(grid, list) or not grid:
        return False

    if not all(isinstance(row, list) and row for row in grid):
        return False

    width = len(grid[0])

    return all(len(row) == width for row in grid)


def grid_shape(grid: Any) -> tuple[int, int]:
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
    width = max((len(row) for row in valid_rows), default=0)

    return height, width


def iter_cells(grid: list[list[int]]) -> Iterable[tuple[int, int, int]]:
    for row_index, row in enumerate(grid):
        for column_index, color in enumerate(row):
            yield row_index, column_index, color


def border_cells(grid: list[list[int]]) -> list[int]:
    if not is_grid(grid):
        return []

    height = len(grid)
    width = len(grid[0])

    if height == 1:
        return list(grid[0])

    if width == 1:
        return [row[0] for row in grid]

    values: list[int] = []
    values.extend(grid[0])
    values.extend(grid[-1])

    for row_index in range(1, height - 1):
        values.append(grid[row_index][0])
        values.append(grid[row_index][-1])

    return values


def neighbors(
    row: int,
    column: int,
    height: int,
    width: int,
    connectivity: int,
) -> Iterable[tuple[int, int]]:
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    if connectivity == 8:
        directions.extend(
            [
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
        )

    for row_change, column_change in directions:
        next_row = row + row_change
        next_column = column + column_change

        if 0 <= next_row < height and 0 <= next_column < width:
            yield next_row, next_column


def connected_component_sizes(
    grid: list[list[int]],
    connectivity: int,
) -> list[int]:
    if not is_grid(grid):
        return []

    height = len(grid)
    width = len(grid[0])
    visited: set[tuple[int, int]] = set()
    component_sizes: list[int] = []

    for start_row in range(height):
        for start_column in range(width):
            start = (start_row, start_column)

            if start in visited:
                continue

            color = grid[start_row][start_column]
            queue: deque[tuple[int, int]] = deque([start])
            visited.add(start)
            size = 0

            while queue:
                row, column = queue.popleft()
                size += 1

                for next_row, next_column in neighbors(
                    row=row,
                    column=column,
                    height=height,
                    width=width,
                    connectivity=connectivity,
                ):
                    next_cell = (next_row, next_column)

                    if next_cell in visited:
                        continue

                    if grid[next_row][next_column] != color:
                        continue

                    visited.add(next_cell)
                    queue.append(next_cell)

            component_sizes.append(size)

    return component_sizes


def repeated_row_pairs(grid: list[list[int]]) -> int:
    if not is_grid(grid):
        return 0

    rows = [tuple(row) for row in grid]
    counts = Counter(rows)

    return sum(count * (count - 1) // 2 for count in counts.values())


def repeated_column_pairs(grid: list[list[int]]) -> int:
    if not is_grid(grid):
        return 0

    height = len(grid)
    width = len(grid[0])

    columns = [
        tuple(grid[row][column] for row in range(height))
        for column in range(width)
    ]

    counts = Counter(columns)

    return sum(count * (count - 1) // 2 for count in counts.values())


def repeated_2x2_count(grid: list[list[int]]) -> int:
    if not is_grid(grid):
        return 0

    height = len(grid)
    width = len(grid[0])

    if height < 2 or width < 2:
        return 0

    patches: list[tuple[int, int, int, int]] = []

    for row in range(height - 1):
        for column in range(width - 1):
            patches.append(
                (
                    grid[row][column],
                    grid[row][column + 1],
                    grid[row + 1][column],
                    grid[row + 1][column + 1],
                )
            )

    counts = Counter(patches)

    return sum(count - 1 for count in counts.values() if count > 1)


def full_line_indexes(
    grid: list[list[int]],
) -> tuple[list[int], list[int]]:
    if not is_grid(grid):
        return [], []

    height = len(grid)
    width = len(grid[0])

    horizontal = [
        row_index
        for row_index, row in enumerate(grid)
        if len(set(row)) == 1
    ]

    vertical = [
        column_index
        for column_index in range(width)
        if len({grid[row][column_index] for row in range(height)}) == 1
    ]

    return horizontal, vertical


# =============================================================================
# RAW GRID MEASUREMENT
# =============================================================================

def measure_grid_facts(grid: Any) -> GridFacts:
    if not is_grid(grid):
        height, width = grid_shape(grid)

        return GridFacts(
            height=height,
            width=width,
            cells=height * width,
            rectangular=False,
            colors=(),
            color_count=0,
            color_counts={},
            most_common_color=None,
            most_common_count=0,
            dominant_fraction=0.0,
            border_counts={},
            most_common_border_color=None,
            border_dominant_fraction=0.0,
            full_horizontal_lines=(),
            full_vertical_lines=(),
            repeated_row_pairs=0,
            repeated_column_pairs=0,
            repeated_2x2_count=0,
            four_connected_component_count=0,
            eight_connected_component_count=0,
            largest_four_component_fraction=0.0,
            largest_eight_component_fraction=0.0,
        )

    height = len(grid)
    width = len(grid[0])
    cells = height * width

    color_counts_counter = Counter(
        color
        for _, _, color in iter_cells(grid)
    )

    colors = tuple(sorted(color_counts_counter))

    most_common_color: int | None = None
    most_common_count = 0

    if color_counts_counter:
        most_common_color, most_common_count = (
            color_counts_counter.most_common(1)[0]
        )

    border_counts_counter = Counter(border_cells(grid))

    most_common_border_color: int | None = None
    most_common_border_count = 0

    if border_counts_counter:
        most_common_border_color, most_common_border_count = (
            border_counts_counter.most_common(1)[0]
        )

    horizontal_lines, vertical_lines = full_line_indexes(grid)

    four_sizes = connected_component_sizes(grid, connectivity=4)
    eight_sizes = connected_component_sizes(grid, connectivity=8)

    return GridFacts(
        height=height,
        width=width,
        cells=cells,
        rectangular=True,
        colors=colors,
        color_count=len(colors),
        color_counts=dict(sorted(color_counts_counter.items())),
        most_common_color=most_common_color,
        most_common_count=most_common_count,
        dominant_fraction=(most_common_count / cells if cells else 0.0),
        border_counts=dict(sorted(border_counts_counter.items())),
        most_common_border_color=most_common_border_color,
        border_dominant_fraction=(
            most_common_border_count / sum(border_counts_counter.values())
            if border_counts_counter
            else 0.0
        ),
        full_horizontal_lines=tuple(horizontal_lines),
        full_vertical_lines=tuple(vertical_lines),
        repeated_row_pairs=repeated_row_pairs(grid),
        repeated_column_pairs=repeated_column_pairs(grid),
        repeated_2x2_count=repeated_2x2_count(grid),
        four_connected_component_count=len(four_sizes),
        eight_connected_component_count=len(eight_sizes),
        largest_four_component_fraction=(
            max(four_sizes, default=0) / cells
            if cells
            else 0.0
        ),
        largest_eight_component_fraction=(
            max(eight_sizes, default=0) / cells
            if cells
            else 0.0
        ),
    )


# =============================================================================
# SIZE GROUPING
# =============================================================================

def measure_task_size(task_id: str, task: dict[str, Any]) -> TaskSize:
    train_pairs = task.get("train", [])

    measurements: list[tuple[int, int, int]] = []

    for pair in train_pairs:
        if not isinstance(pair, dict):
            continue

        height, width = grid_shape(pair.get("input", []))
        measurements.append((height, width, height * width))

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


def measure_all_task_sizes(data: dict[str, Any]) -> list[TaskSize]:
    measurements = [
        measure_task_size(task_id, task)
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


def find_natural_breaks(
    measurements: list[TaskSize],
    group_count: int,
) -> list[int]:
    task_count = len(measurements)

    if task_count <= 1:
        return []

    desired_break_count = min(group_count - 1, task_count - 1)
    gap_candidates: list[tuple[int, int]] = []

    for index in range(task_count - 1):
        current_cells = measurements[index].largest_cells
        next_cells = measurements[index + 1].largest_cells
        gap = next_cells - current_cells

        if gap > 0:
            gap_candidates.append((gap, index + 1))

    gap_candidates.sort(key=lambda item: (-item[0], item[1]))

    chosen_breaks = [
        break_index
        for _, break_index in gap_candidates[:desired_break_count]
    ]

    if len(chosen_breaks) < desired_break_count:
        for group_number in range(1, group_count):
            suggested_index = round(task_count * group_number / group_count)
            suggested_index = max(1, min(suggested_index, task_count - 1))

            if suggested_index not in chosen_breaks:
                chosen_breaks.append(suggested_index)

            if len(chosen_breaks) >= desired_break_count:
                break

    return sorted(set(chosen_breaks))[:desired_break_count]


def split_size_groups(
    measurements: list[TaskSize],
) -> dict[str, list[TaskSize]]:
    breaks = find_natural_breaks(
        measurements=measurements,
        group_count=len(SIZE_GROUP_NAMES),
    )

    boundaries = [0, *breaks, len(measurements)]

    groups: dict[str, list[TaskSize]] = {
        group_name: []
        for group_name in SIZE_GROUP_NAMES
    }

    for group_index, group_name in enumerate(SIZE_GROUP_NAMES):
        start = boundaries[group_index]
        end = boundaries[group_index + 1]
        groups[group_name] = measurements[start:end]

    return groups


# =============================================================================
# OUTPUT SIZE RELATIONSHIP GROUPING
# =============================================================================

def observe_pair_output_size(input_grid: Any, output_grid: Any) -> str:
    input_height, input_width = grid_shape(input_grid)
    output_height, output_width = grid_shape(output_grid)

    if min(input_height, input_width, output_height, output_width) == 0:
        return "invalid_grid"

    if input_height == output_height and input_width == output_width:
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


def observe_task_output_size(task: dict[str, Any]) -> tuple[str, list[str]]:
    train_pairs = task.get("train", [])

    if not isinstance(train_pairs, list) or not train_pairs:
        return "no_train_pairs", []

    pair_groups = [
        observe_pair_output_size(
            pair.get("input") if isinstance(pair, dict) else None,
            pair.get("output") if isinstance(pair, dict) else None,
        )
        for pair in train_pairs
    ]

    unique_groups = sorted(set(pair_groups))

    if len(unique_groups) == 1:
        return unique_groups[0], pair_groups

    return "mixed__" + "__".join(unique_groups), pair_groups


# =============================================================================
# BACKGROUND EVIDENCE
# =============================================================================

def score_background_candidates(
    facts_list: list[GridFacts],
) -> tuple[str, dict[str, float], list[int]]:
    if not facts_list:
        return "uncertain", {}, []

    all_colors = sorted(
        {
            color
            for facts in facts_list
            for color in facts.colors
        }
    )

    color_scores: dict[int, float] = defaultdict(float)

    for facts in facts_list:
        if facts.cells == 0:
            continue

        for color in facts.colors:
            cell_fraction = facts.color_counts.get(color, 0) / facts.cells
            border_total = sum(facts.border_counts.values())
            border_fraction = (
                facts.border_counts.get(color, 0) / border_total
                if border_total
                else 0.0
            )

            score = 55.0 * cell_fraction + 35.0 * border_fraction

            if color == facts.most_common_color:
                score += 8.0

            if color == facts.most_common_border_color:
                score += 12.0

            color_scores[color] += score

    pair_count = max(1, len(facts_list))

    averaged_scores = {
        str(color): round(score / pair_count, 3)
        for color, score in color_scores.items()
    }

    ranked = sorted(
        color_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    if not ranked:
        return "uncertain", averaged_scores, []

    best_color, best_total = ranked[0]
    best_score = best_total / pair_count
    second_score = (
        ranked[1][1] / pair_count
        if len(ranked) > 1
        else 0.0
    )
    margin = best_score - second_score

    dominant_fractions = [facts.dominant_fraction for facts in facts_list]
    border_fractions = [facts.border_dominant_fraction for facts in facts_list]

    average_dominant = sum(dominant_fractions) / len(dominant_fractions)
    average_border = sum(border_fractions) / len(border_fractions)

    winners = [
        facts.most_common_color
        for facts in facts_list
        if facts.most_common_color is not None
    ]
    winner_consistency = (
        Counter(winners).most_common(1)[0][1] / len(winners)
        if winners
        else 0.0
    )

    candidates = [best_color]

    if len(ranked) > 1 and margin < 10.0:
        candidates.append(ranked[1][0])

    if average_dominant < 0.32 and average_border < 0.38:
        group = "none"
    elif len(candidates) > 1 and margin < 6.0:
        group = "multiple"
    elif (
        average_dominant >= 0.55
        and average_border >= 0.55
        and winner_consistency >= 0.67
        and margin >= 12.0
    ):
        group = "clear"
    elif best_score >= 45.0 and winner_consistency >= 0.5:
        group = "possible"
    else:
        group = "uncertain"

    background_scores = {
        "best_color_score": round(best_score, 3),
        "second_color_score": round(second_score, 3),
        "score_margin": round(margin, 3),
        "average_dominant_fraction": round(average_dominant, 3),
        "average_border_dominant_fraction": round(average_border, 3),
        "winner_consistency": round(winner_consistency, 3),
        "per_color": averaged_scores,
    }

    return group, background_scores, candidates


# =============================================================================
# SCENE EVIDENCE
# =============================================================================

def score_scene(
    input_facts: list[GridFacts],
    background_group: str,
) -> tuple[str, dict[str, float]]:
    if not input_facts:
        return "uncertain", {
            "object": 0.0,
            "pattern": 0.0,
            "mixed": 0.0,
        }

    object_score = 0.0
    pattern_score = 0.0

    for facts in input_facts:
        if facts.cells == 0:
            continue

        repetition_density = min(
            1.0,
            (
                facts.repeated_row_pairs
                + facts.repeated_column_pairs
                + facts.repeated_2x2_count
            )
            / max(1, facts.cells),
        )

        component_density = min(
            1.0,
            facts.four_connected_component_count / max(1, facts.cells),
        )

        line_count = (
            len(facts.full_horizontal_lines)
            + len(facts.full_vertical_lines)
        )

        pattern_score += 45.0 * repetition_density
        pattern_score += 20.0 * min(1.0, facts.color_count / 8.0)
        pattern_score += 18.0 * max(0.0, 0.45 - facts.dominant_fraction) / 0.45
        pattern_score += 10.0 * min(1.0, line_count / 3.0)
        pattern_score += 7.0 * component_density

        object_score += 35.0 * facts.dominant_fraction
        object_score += 20.0 * facts.border_dominant_fraction
        object_score += 18.0 * facts.largest_four_component_fraction
        object_score += 12.0 * min(1.0, facts.four_connected_component_count / 12.0)
        object_score += 10.0 * min(1.0, line_count / 3.0)
        object_score += 5.0 * max(0.0, 1.0 - facts.color_count / 10.0)

    pair_count = len(input_facts)
    object_score /= pair_count
    pattern_score /= pair_count

    if background_group == "clear":
        object_score += 18.0
    elif background_group == "possible":
        object_score += 9.0
    elif background_group == "none":
        pattern_score += 18.0
    elif background_group == "multiple":
        pattern_score += 6.0
        object_score += 6.0

    difference = abs(object_score - pattern_score)
    mixed_score = min(object_score, pattern_score) + max(0.0, 20.0 - difference)

    scores = {
        "object": round(object_score, 3),
        "pattern": round(pattern_score, 3),
        "mixed": round(mixed_score, 3),
    }

    best_name, best_score = max(scores.items(), key=lambda item: item[1])
    ordered_scores = sorted(scores.values(), reverse=True)
    margin = ordered_scores[0] - ordered_scores[1]

    if best_score < 32.0:
        group = "uncertain"
    elif margin < 5.0:
        group = "mixed"
    else:
        group = best_name

    return group, scores


# =============================================================================
# STRUCTURE AND MEASUREMENT TAGS
# =============================================================================

def build_structure_tags(
    input_facts: list[GridFacts],
    output_facts: list[GridFacts],
    background_group: str,
    scene_group: str,
    output_size_group: str,
) -> list[str]:
    tags: set[str] = set()

    all_facts = [*input_facts, *output_facts]

    if scene_group == "pattern":
        tags.add("pattern_scene")
    elif scene_group == "object":
        tags.add("object_scene")
    elif scene_group == "mixed":
        tags.add("mixed_scene")

    if background_group == "none":
        tags.add("no_clear_background")
    elif background_group == "clear":
        tags.add("clear_background")
    elif background_group == "multiple":
        tags.add("multiple_background_candidates")

    if any(facts.repeated_2x2_count > 0 for facts in all_facts):
        tags.add("repeated_local_patches")

    if any(facts.repeated_row_pairs > 0 for facts in all_facts):
        tags.add("repeated_rows")

    if any(facts.repeated_column_pairs > 0 for facts in all_facts):
        tags.add("repeated_columns")

    if any(facts.full_horizontal_lines for facts in all_facts):
        tags.add("full_horizontal_lines")

    if any(facts.full_vertical_lines for facts in all_facts):
        tags.add("full_vertical_lines")

    if any(
        facts.four_connected_component_count
        != facts.eight_connected_component_count
        for facts in all_facts
    ):
        tags.add("diagonal_connectivity_changes_objects")

    if any(facts.four_connected_component_count >= 10 for facts in input_facts):
        tags.add("many_color_components")

    if any(facts.color_count >= 7 for facts in input_facts):
        tags.add("many_colors")

    if any(facts.color_count <= 3 for facts in input_facts):
        tags.add("few_colors")

    if output_size_group == "same_size":
        tags.add("same_size_output")
    elif "grow" in output_size_group:
        tags.add("output_grows")
    elif "shrink" in output_size_group:
        tags.add("output_shrinks")
    elif output_size_group.startswith("mixed__"):
        tags.add("mixed_output_sizes")

    return sorted(tags)


def build_measurement_tags(
    input_facts: list[GridFacts],
) -> list[str]:
    tags: set[str] = set()

    if not input_facts:
        return []

    if all(facts.height == facts.width for facts in input_facts):
        tags.add("square_inputs")

    if all(facts.width > facts.height for facts in input_facts):
        tags.add("wide_inputs")

    if all(facts.height > facts.width for facts in input_facts):
        tags.add("tall_inputs")

    if len({(facts.height, facts.width) for facts in input_facts}) == 1:
        tags.add("fixed_train_input_size")
    else:
        tags.add("variable_train_input_size")

    if len({facts.color_count for facts in input_facts}) == 1:
        tags.add("fixed_train_color_count")
    else:
        tags.add("variable_train_color_count")

    if all(facts.rectangular for facts in input_facts):
        tags.add("rectangular_grids")

    return sorted(tags)


# =============================================================================
# COMPLETE TASK CLASSIFICATION
# =============================================================================

def classify_task(
    task_id: str,
    task: dict[str, Any],
    task_size: TaskSize,
    size_group: str,
) -> TaskClassification:
    train_pairs = task.get("train", [])

    input_facts: list[GridFacts] = []
    output_facts: list[GridFacts] = []

    for pair in train_pairs:
        if not isinstance(pair, dict):
            continue

        input_facts.append(measure_grid_facts(pair.get("input")))
        output_facts.append(measure_grid_facts(pair.get("output")))

    output_size_group, pair_output_size_groups = observe_task_output_size(task)

    background_group, background_scores, background_candidates = (
        score_background_candidates(input_facts)
    )

    scene_group, scene_scores = score_scene(
        input_facts=input_facts,
        background_group=background_group,
    )

    structure_tags = build_structure_tags(
        input_facts=input_facts,
        output_facts=output_facts,
        background_group=background_group,
        scene_group=scene_group,
        output_size_group=output_size_group,
    )

    measurement_tags = build_measurement_tags(input_facts)

    return TaskClassification(
        task_id=task_id,
        size_group=size_group,
        output_size_group=output_size_group,
        train_pair_output_size_groups=pair_output_size_groups,
        scene_group=scene_group,
        scene_scores=scene_scores,
        background_group=background_group,
        background_scores=background_scores,
        background_candidates=background_candidates,
        structure_tags=structure_tags,
        measurement_tags=measurement_tags,
        largest_train_input={
            "height": task_size.largest_height,
            "width": task_size.largest_width,
            "cells": task_size.largest_cells,
        },
        smallest_train_input={
            "height": task_size.smallest_height,
            "width": task_size.smallest_width,
            "cells": task_size.smallest_cells,
        },
        train_input_count=task_size.train_input_count,
        train_input_facts=[asdict(facts) for facts in input_facts],
        train_output_facts=[asdict(facts) for facts in output_facts],
    )


# =============================================================================
# GROUP BUILDING
# =============================================================================

def add_task_to_group(
    groups: dict[str, dict[str, Any]],
    group_name: str,
    task_id: str,
    task: dict[str, Any],
) -> None:
    groups.setdefault(group_name, {})[task_id] = task


def build_all_groups(
    data: dict[str, Any],
    classifications: dict[str, TaskClassification],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {
        "size": {},
        "scene": {},
        "background": {},
        "output_size": {},
        "structure": {},
        "measurement": {},
    }

    for task_id, classification in classifications.items():
        task = data[task_id]

        add_task_to_group(
            grouped["size"],
            classification.size_group,
            task_id,
            task,
        )

        add_task_to_group(
            grouped["scene"],
            classification.scene_group,
            task_id,
            task,
        )

        add_task_to_group(
            grouped["background"],
            classification.background_group,
            task_id,
            task,
        )

        add_task_to_group(
            grouped["output_size"],
            classification.output_size_group,
            task_id,
            task,
        )

        for tag in classification.structure_tags or []:
            add_task_to_group(
                grouped["structure"],
                tag,
                task_id,
                task,
            )

        for tag in classification.measurement_tags or []:
            add_task_to_group(
                grouped["measurement"],
                tag,
                task_id,
                task,
            )

    return grouped


# =============================================================================
# SAVING
# =============================================================================

def prepare_output_folder(output_folder: Path) -> None:
    if output_folder.exists():
        shutil.rmtree(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)


def save_json(file_path: Path, data: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_text(file_path: Path, lines: list[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("\n".join(lines), encoding="utf-8")


def save_group_files(
    grouped: dict[str, dict[str, dict[str, Any]]],
    output_folder: Path,
) -> None:
    for category_name, category_groups in grouped.items():
        category_folder = output_folder / category_name

        for group_name, tasks in sorted(category_groups.items()):
            save_json(
                category_folder / f"{group_name}.json",
                dict(sorted(tasks.items())),
            )


def save_legacy_size_files(
    grouped: dict[str, dict[str, dict[str, Any]]],
    output_folder: Path,
) -> None:
    """
    Keep the original top-level size files so the current viewers and scripts
    do not break while all new grouping lives in category folders.
    """
    size_groups = grouped["size"]

    for group_name in SIZE_GROUP_NAMES:
        save_json(
            output_folder / f"{group_name}.json",
            dict(sorted(size_groups.get(group_name, {}).items())),
        )


def save_task_index(
    classifications: dict[str, TaskClassification],
    output_folder: Path,
) -> None:
    index = {
        "description": (
            "One factual classification record per task. A task can belong "
            "to many independent groups at the same time."
        ),
        "tasks": {
            task_id: asdict(classification)
            for task_id, classification in sorted(classifications.items())
        },
    }

    save_json(output_folder / "task_index.json", index)


def save_group_inventory(
    grouped: dict[str, dict[str, dict[str, Any]]],
    output_folder: Path,
) -> None:
    inventory: dict[str, Any] = {"categories": {}}

    for category_name, category_groups in grouped.items():
        inventory["categories"][category_name] = {}

        for group_name, tasks in sorted(category_groups.items()):
            inventory["categories"][category_name][group_name] = {
                "task_count": len(tasks),
                "task_ids": sorted(tasks),
            }

    save_json(output_folder / "group_inventory.json", inventory)


# =============================================================================
# REPORTING
# =============================================================================

def build_report(
    grouped: dict[str, dict[str, dict[str, Any]]],
    classifications: dict[str, TaskClassification],
) -> list[str]:
    lines = [
        "ARCs5 COMPLETE FIRST-LOOK GROUPING",
        "=" * 88,
        "",
        f"TOTAL TASKS: {len(classifications)}",
        "",
    ]

    for category_name, category_groups in grouped.items():
        lines.append(category_name.upper().replace("_", " "))
        lines.append("-" * 88)

        for group_name, tasks in sorted(
            category_groups.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ):
            lines.append(f"{group_name:<48} {len(tasks):>3} tasks")

        lines.append("")

    lines.append("TASK CLASSIFICATIONS")
    lines.append("-" * 88)

    for task_id, classification in sorted(classifications.items()):
        structure_text = ", ".join(classification.structure_tags or [])

        lines.append(
            f"{task_id} | "
            f"size={classification.size_group:<14} | "
            f"scene={classification.scene_group:<9} | "
            f"background={classification.background_group:<9} | "
            f"output={classification.output_size_group}"
        )

        if structure_text:
            lines.append(f"             structure: {structure_text}")

    lines.append("")

    return lines


def print_live_classification(
    number: int,
    total: int,
    classification: TaskClassification,
) -> None:
    largest = classification.largest_train_input or {}
    shape = f"{largest.get('height', 0)}x{largest.get('width', 0)}"

    print(
        f"[{number:03}/{total:03}] "
        f"{classification.task_id} | "
        f"{shape:<7} | "
        f"{classification.size_group:<14} | "
        f"{classification.scene_group:<9} | "
        f"bg={classification.background_group:<9} | "
        f"out={classification.output_size_group}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    data = load_data(DATA_FILE)

    task_sizes = measure_all_task_sizes(data)

    if not task_sizes:
        print("No tasks were found in data.json.")
        return

    size_groups = split_size_groups(task_sizes)

    task_to_size_group: dict[str, str] = {}

    for group_name, tasks in size_groups.items():
        for task_size in tasks:
            task_to_size_group[task_size.task_id] = group_name

    size_by_task_id = {
        task_size.task_id: task_size
        for task_size in task_sizes
    }

    prepare_output_folder(OUTPUT_FOLDER)

    print()
    print("ARCs5 COMPLETE FIRST-LOOK GROUPING")
    print("=" * 88)

    classifications: dict[str, TaskClassification] = {}
    task_count = len(data)

    for number, task_id in enumerate(sorted(data), start=1):
        classification = classify_task(
            task_id=task_id,
            task=data[task_id],
            task_size=size_by_task_id[task_id],
            size_group=task_to_size_group.get(task_id, "unknown"),
        )

        classifications[task_id] = classification

        print_live_classification(
            number=number,
            total=task_count,
            classification=classification,
        )

    grouped = build_all_groups(
        data=data,
        classifications=classifications,
    )

    save_group_files(
        grouped=grouped,
        output_folder=OUTPUT_FOLDER,
    )

    save_legacy_size_files(
        grouped=grouped,
        output_folder=OUTPUT_FOLDER,
    )

    save_task_index(
        classifications=classifications,
        output_folder=OUTPUT_FOLDER,
    )

    save_group_inventory(
        grouped=grouped,
        output_folder=OUTPUT_FOLDER,
    )

    report = build_report(
        grouped=grouped,
        classifications=classifications,
    )

    save_text(
        OUTPUT_FOLDER / "group_report.txt",
        report,
    )

    print()
    print("\n".join(report))

    print("FILES CREATED")
    print("-" * 88)
    print("Top-level compatibility size files:")

    for group_name in SIZE_GROUP_NAMES:
        task_total = len(grouped["size"].get(group_name, {}))
        print(f"  {group_name + '.json':<28} {task_total:>3} tasks")

    print()
    print("Category folders:")

    for category_name, category_groups in grouped.items():
        print(
            f"  {category_name + '/':<28} "
            f"{len(category_groups):>3} subgroup files"
        )

    print()
    print("Master files:")
    print("  task_index.json")
    print("  group_inventory.json")
    print("  group_report.txt")
    print()
    print(f"Output folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()