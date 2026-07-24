from collections import Counter
from typing import Any


Grid = list[list[int]]


def get_border_cells(grid: Grid) -> list[int]:
    """
    Return each outer-border cell exactly once.
    """
    height = len(grid)
    width = len(grid[0]) if height else 0

    if height == 0 or width == 0:
        return []

    if height == 1:
        return list(grid[0])

    if width == 1:
        return [row[0] for row in grid]

    border_cells = []

    border_cells.extend(grid[0])
    border_cells.extend(grid[-1])

    for row_index in range(1, height - 1):
        border_cells.append(grid[row_index][0])
        border_cells.append(grid[row_index][-1])

    return border_cells


def observe_colors(grid: Grid) -> dict[str, Any]:
    """
    Observe color usage and make a cautious background estimate.

    A background is only accepted when one color occupies at least
    half of the complete grid.
    """
    flattened = [
        color
        for row in grid
        for color in row
    ]

    if not flattened:
        return {
            "colors": [],
            "color_count": 0,
            "color_counts": {},
            "most_common_color": None,
            "dominant_fraction": 0.0,
            "border_counts": {},
            "most_common_border": None,
            "background_confident": False,
            "background": None,
            "grid_type": "empty",
        }

    color_counts = Counter(flattened)
    colors = sorted(color_counts)

    most_common_color, most_common_count = color_counts.most_common(1)[0]

    dominant_fraction = most_common_count / len(flattened)

    border_cells = get_border_cells(grid)
    border_counts = Counter(border_cells)

    most_common_border = (
        border_counts.most_common(1)[0][0]
        if border_counts
        else None
    )

    background_confident = dominant_fraction >= 0.50

    background = (
        most_common_color
        if background_confident
        else None
    )

    if background_confident:
        grid_type = "background_with_foreground"
    else:
        grid_type = "dense_pattern"

    return {
        "colors": colors,
        "color_count": len(colors),
        "color_counts": dict(sorted(color_counts.items())),
        "most_common_color": most_common_color,
        "dominant_fraction": round(dominant_fraction, 3),
        "border_counts": dict(sorted(border_counts.items())),
        "most_common_border": most_common_border,
        "background_confident": background_confident,
        "background": background,
        "grid_type": grid_type,
    }


def task_color_group(task: dict) -> str:
    counts = []

    for pair in task.get("train", []):
        observation = observe_colors(pair.get("input", []))
        counts.append(observation["color_count"])

    unique_counts = sorted(set(counts))

    if not unique_counts:
        return "colors_0"

    if len(unique_counts) == 1:
        return f"colors_{unique_counts[0]}"

    joined = "_".join(str(count) for count in unique_counts)
    return f"mixed_colors_{joined}"