from typing import Any


Grid = list[list[int]]


def observe_lines(grid: Grid) -> dict[str, Any]:
    """
    Detect complete single-color rows and columns.

    These may represent dividers, borders, bars, or separators.
    This observer only reports what exists. It does not decide meaning.
    """
    height = len(grid)
    width = len(grid[0]) if height else 0

    full_horizontal_lines = []
    full_vertical_lines = []

    for row_index, row in enumerate(grid):
        if row and len(set(row)) == 1:
            full_horizontal_lines.append({
                "row": row_index,
                "color": row[0],
                "length": len(row),
            })

    for column_index in range(width):
        column = [
            grid[row_index][column_index]
            for row_index in range(height)
        ]

        if column and len(set(column)) == 1:
            full_vertical_lines.append({
                "column": column_index,
                "color": column[0],
                "length": len(column),
            })

    return {
        "full_horizontal_line_count": len(full_horizontal_lines),
        "full_horizontal_lines": full_horizontal_lines,
        "full_vertical_line_count": len(full_vertical_lines),
        "full_vertical_lines": full_vertical_lines,
        "has_full_line": bool(
            full_horizontal_lines or full_vertical_lines
        ),
    }