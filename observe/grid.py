from typing import Any


Grid = list[list[int]]


def observe_grid(grid: Grid) -> dict[str, Any]:
    """
    Observe basic facts about the grid itself.

    This module does not inspect colors, objects, shapes,
    transformations, or task rules.
    """
    height = len(grid)
    width = len(grid[0]) if height else 0

    row_lengths = [len(row) for row in grid]
    rectangular = all(length == width for length in row_lengths)

    total_cells = sum(row_lengths)

    if height == width:
        shape_type = "square"
    elif height > width:
        shape_type = "tall"
    else:
        shape_type = "wide"

    return {
        "grid_height": height,
        "grid_width": width,
        "grid_total_cells": total_cells,
        "grid_rectangular": rectangular,
        "grid_shape_type": shape_type,
    }
