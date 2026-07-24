from typing import Any

from observe.colors import observe_colors
from observe.grid import observe_grid
from observe.lines import observe_lines


Grid = list[list[int]]


def observe(grid: Grid) -> dict[str, Any]:
    """
    Run every active observer and combine the results.
    """
    observations: dict[str, Any] = {}

    observations.update(observe_grid(grid))
    observations.update(observe_colors(grid))
    observations.update(observe_lines(grid))

    return observations