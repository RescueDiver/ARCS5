from __future__ import annotations

from collections import deque
from typing import Any


Grid = list[list[int]]
Cell = tuple[int, int]


# =============================================================================
# GRID HELPERS
# =============================================================================

def grid_dimensions(
    grid: Grid,
) -> tuple[int, int]:
    if not grid:
        return 0, 0

    return len(grid), len(grid[0])


def is_rectangular(
    grid: Grid,
) -> bool:
    if not grid:
        return True

    expected_width = len(grid[0])

    return all(
        len(row) == expected_width
        for row in grid
    )


def is_inside_grid(
    row: int,
    column: int,
    height: int,
    width: int,
) -> bool:
    return (
        0 <= row < height
        and 0 <= column < width
    )


def get_neighbors(
    row: int,
    column: int,
    connectivity: int,
) -> tuple[Cell, ...]:
    """
    Return neighboring cells using either:

    4-connectivity:
        up, down, left, right

    8-connectivity:
        up, down, left, right, and diagonals
    """

    four_neighbors = (
        (row - 1, column),
        (row + 1, column),
        (row, column - 1),
        (row, column + 1),
    )

    if connectivity == 4:
        return four_neighbors

    if connectivity == 8:
        return four_neighbors + (
            (row - 1, column - 1),
            (row - 1, column + 1),
            (row + 1, column - 1),
            (row + 1, column + 1),
        )

    raise ValueError(
        f"Unsupported connectivity: {connectivity}"
    )


# =============================================================================
# CONNECTED COMPONENT
# =============================================================================

def collect_connected_component(
    grid: Grid,
    start_row: int,
    start_column: int,
    visited: set[Cell],
    connectivity: int,
) -> list[Cell]:
    """
    Collect one connected component of one color.

    The connectivity determines whether diagonal contact joins cells.
    """

    height, width = grid_dimensions(grid)
    object_color = grid[start_row][start_column]

    cells: list[Cell] = []

    queue: deque[Cell] = deque()
    queue.append((start_row, start_column))

    visited.add((start_row, start_column))

    while queue:
        row, column = queue.popleft()
        cells.append((row, column))

        for neighbor_row, neighbor_column in get_neighbors(
            row=row,
            column=column,
            connectivity=connectivity,
        ):
            if not is_inside_grid(
                row=neighbor_row,
                column=neighbor_column,
                height=height,
                width=width,
            ):
                continue

            neighbor = (
                neighbor_row,
                neighbor_column,
            )

            if neighbor in visited:
                continue

            if (
                grid[neighbor_row][neighbor_column]
                != object_color
            ):
                continue

            visited.add(neighbor)
            queue.append(neighbor)

    cells.sort()

    return cells


# =============================================================================
# OBJECT FACTS
# =============================================================================

def build_object_record(
    object_id: int,
    color: int,
    cells: list[Cell],
    grid_height: int,
    grid_width: int,
    connectivity: int,
) -> dict[str, Any]:
    rows = [
        row
        for row, _ in cells
    ]

    columns = [
        column
        for _, column in cells
    ]

    top = min(rows)
    bottom = max(rows)
    left = min(columns)
    right = max(columns)

    height = bottom - top + 1
    width = right - left + 1

    pixel_count = len(cells)
    bounding_box_area = height * width

    touches_top = top == 0
    touches_bottom = bottom == grid_height - 1
    touches_left = left == 0
    touches_right = right == grid_width - 1

    touches_border = any(
        (
            touches_top,
            touches_bottom,
            touches_left,
            touches_right,
        )
    )

    normalized_cells = [
        (
            row - top,
            column - left,
        )
        for row, column in cells
    ]

    return {
        "id": object_id,
        "connectivity": connectivity,
        "color": color,
        "cells": cells,
        "normalized_cells": normalized_cells,
        "pixel_count": pixel_count,
        "bbox": {
            "top": top,
            "left": left,
            "bottom": bottom,
            "right": right,
        },
        "height": height,
        "width": width,
        "bounding_box_area": bounding_box_area,
        "fills_bounding_box": (
            pixel_count == bounding_box_area
        ),
        "touches_border": touches_border,
        "touches_top": touches_top,
        "touches_bottom": touches_bottom,
        "touches_left": touches_left,
        "touches_right": touches_right,
    }


# =============================================================================
# ONE OBJECT VIEW
# =============================================================================

def find_color_objects(
    grid: Grid,
    background: int,
    connectivity: int,
) -> list[dict[str, Any]]:
    """
    Find same-color connected components using the requested connectivity.
    """

    height, width = grid_dimensions(grid)

    visited: set[Cell] = set()
    objects: list[dict[str, Any]] = []

    for row in range(height):
        for column in range(width):
            cell = (row, column)
            color = grid[row][column]

            if cell in visited:
                continue

            if color == background:
                visited.add(cell)
                continue

            cells = collect_connected_component(
                grid=grid,
                start_row=row,
                start_column=column,
                visited=visited,
                connectivity=connectivity,
            )

            object_record = build_object_record(
                object_id=len(objects) + 1,
                color=color,
                cells=cells,
                grid_height=height,
                grid_width=width,
                connectivity=connectivity,
            )

            objects.append(object_record)

    return objects


# =============================================================================
# OBJECT OBSERVER
# =============================================================================

def observe_objects(
    grid: Grid,
    background: int | None,
) -> dict[str, Any]:
    """
    Return multiple candidate interpretations of same-color objects.

    color_4_objects:
        Cells join only through shared sides.

    color_8_objects:
        Cells join through shared sides or diagonal contact.

    Neither view is declared to be the final or correct interpretation.
    """

    if not grid:
        return {
            "object_detection_available": False,
            "object_detection_reason": "empty_grid",
            "color_4_object_count": 0,
            "color_4_objects": [],
            "color_8_object_count": 0,
            "color_8_objects": [],
        }

    if not is_rectangular(grid):
        return {
            "object_detection_available": False,
            "object_detection_reason": "non_rectangular_grid",
            "color_4_object_count": 0,
            "color_4_objects": [],
            "color_8_object_count": 0,
            "color_8_objects": [],
        }

    if background is None:
        return {
            "object_detection_available": False,
            "object_detection_reason": (
                "background_not_confident"
            ),
            "color_4_object_count": 0,
            "color_4_objects": [],
            "color_8_object_count": 0,
            "color_8_objects": [],
        }

    color_4_objects = find_color_objects(
        grid=grid,
        background=background,
        connectivity=4,
    )

    color_8_objects = find_color_objects(
        grid=grid,
        background=background,
        connectivity=8,
    )

    return {
        "object_detection_available": True,
        "object_detection_reason": None,
        "object_candidate_views": [
            "same_color_4_connected",
            "same_color_8_connected",
        ],
        "color_4_object_count": len(color_4_objects),
        "color_4_objects": color_4_objects,
        "color_8_object_count": len(color_8_objects),
        "color_8_objects": color_8_objects,
    }