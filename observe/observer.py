from typing import Any
from observe.objects import observe_objects
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
    object_observations = observe_objects(
        grid=grid,
        background=observations.get("background"),
    )

    observations.update(object_observations)

    return observations