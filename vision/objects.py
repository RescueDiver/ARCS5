from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


# =============================================================================
# DATA TYPES
# =============================================================================

Cell = tuple[int, int]


@dataclass(frozen=True)
class BoundingBox:
    top: int
    left: int
    bottom: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def area(self) -> int:
        return self.height * self.width


@dataclass(frozen=True)
class VisualObject:
    object_id: int
    color: int
    connectivity: int

    cells: tuple[Cell, ...]
    cell_count: int

    bbox: BoundingBox
    height: int
    width: int
    bbox_area: int
    fill_fraction: float

    touches_top: bool
    touches_bottom: bool
    touches_left: bool
    touches_right: bool
    touches_border: bool

    is_single_cell: bool
    is_horizontal_line: bool
    is_vertical_line: bool
    is_solid_rectangle: bool

    local_mask: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = asdict(self.bbox)
        return data


@dataclass(frozen=True)
class ObjectDetectionResult:
    height: int
    width: int
    connectivity: int
    object_count: int
    objects: tuple[VisualObject, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "connectivity": self.connectivity,
            "object_count": self.object_count,
            "objects": [
                visual_object.to_dict()
                for visual_object in self.objects
            ],
        }


# =============================================================================
# GRID VALIDATION
# =============================================================================

def is_grid(grid: Any) -> bool:
    if not isinstance(grid, list) or not grid:
        return False

    if not all(isinstance(row, list) and row for row in grid):
        return False

    width = len(grid[0])

    if width == 0:
        return False

    return all(len(row) == width for row in grid)


def validate_grid(grid: Any) -> list[list[int]]:
    if not is_grid(grid):
        raise ValueError(
            "Expected a non-empty rectangular grid stored as list[list[int]]."
        )

    for row_index, row in enumerate(grid):
        for column_index, value in enumerate(row):
            if not isinstance(value, int):
                raise ValueError(
                    "Grid cells must be integers. "
                    f"Found {value!r} at row={row_index}, column={column_index}."
                )

    return grid


# =============================================================================
# CONNECTIVITY
# =============================================================================

def neighbor_directions(connectivity: int) -> tuple[Cell, ...]:
    if connectivity == 4:
        return (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        )

    if connectivity == 8:
        return (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )

    raise ValueError("connectivity must be either 4 or 8.")


def iter_neighbors(
    row: int,
    column: int,
    height: int,
    width: int,
    connectivity: int,
) -> Iterable[Cell]:
    for row_change, column_change in neighbor_directions(connectivity):
        next_row = row + row_change
        next_column = column + column_change

        if 0 <= next_row < height and 0 <= next_column < width:
            yield next_row, next_column


# =============================================================================
# OBJECT MEASUREMENT
# =============================================================================

def calculate_bounding_box(cells: Sequence[Cell]) -> BoundingBox:
    if not cells:
        raise ValueError("Cannot calculate a bounding box for zero cells.")

    rows = [row for row, _ in cells]
    columns = [column for _, column in cells]

    return BoundingBox(
        top=min(rows),
        left=min(columns),
        bottom=max(rows),
        right=max(columns),
    )


def build_local_mask(
    cells: Sequence[Cell],
    bbox: BoundingBox,
) -> tuple[tuple[int, ...], ...]:
    occupied = set(cells)

    return tuple(
        tuple(
            1
            if (bbox.top + local_row, bbox.left + local_column) in occupied
            else 0
            for local_column in range(bbox.width)
        )
        for local_row in range(bbox.height)
    )


def build_visual_object(
    object_id: int,
    color: int,
    cells: Sequence[Cell],
    grid_height: int,
    grid_width: int,
    connectivity: int,
) -> VisualObject:
    ordered_cells = tuple(sorted(cells))
    bbox = calculate_bounding_box(ordered_cells)

    cell_count = len(ordered_cells)
    bbox_area = bbox.area
    fill_fraction = cell_count / bbox_area if bbox_area else 0.0

    touches_top = any(row == 0 for row, _ in ordered_cells)
    touches_bottom = any(
        row == grid_height - 1
        for row, _ in ordered_cells
    )
    touches_left = any(column == 0 for _, column in ordered_cells)
    touches_right = any(
        column == grid_width - 1
        for _, column in ordered_cells
    )

    is_single_cell = cell_count == 1
    is_horizontal_line = bbox.height == 1 and cell_count == bbox.width
    is_vertical_line = bbox.width == 1 and cell_count == bbox.height
    is_solid_rectangle = cell_count == bbox_area

    return VisualObject(
        object_id=object_id,
        color=color,
        connectivity=connectivity,
        cells=ordered_cells,
        cell_count=cell_count,
        bbox=bbox,
        height=bbox.height,
        width=bbox.width,
        bbox_area=bbox_area,
        fill_fraction=round(fill_fraction, 6),
        touches_top=touches_top,
        touches_bottom=touches_bottom,
        touches_left=touches_left,
        touches_right=touches_right,
        touches_border=(
            touches_top
            or touches_bottom
            or touches_left
            or touches_right
        ),
        is_single_cell=is_single_cell,
        is_horizontal_line=is_horizontal_line,
        is_vertical_line=is_vertical_line,
        is_solid_rectangle=is_solid_rectangle,
        local_mask=build_local_mask(ordered_cells, bbox),
    )


# =============================================================================
# OBJECT DETECTION
# =============================================================================

def detect_objects(
    grid: Any,
    connectivity: int = 4,
    include_colors: Iterable[int] | None = None,
    exclude_colors: Iterable[int] | None = None,
    minimum_cell_count: int = 1,
) -> ObjectDetectionResult:
    """
    Detect every same-color connected component in a grid.

    Nothing is assumed to be background unless its color is explicitly passed
    through exclude_colors.

    Parameters
    ----------
    grid:
        Rectangular ARC grid stored as list[list[int]].

    connectivity:
        4 for edge-touching components.
        8 for edge-or-corner-touching components.

    include_colors:
        Optional collection of colors to inspect.
        None means inspect every color.

    exclude_colors:
        Optional collection of colors to ignore.
        Useful later when a background color has been verified.

    minimum_cell_count:
        Ignore components smaller than this number of cells.

    Returns
    -------
    ObjectDetectionResult
        Grid dimensions plus all detected VisualObject records.
    """
    validated_grid = validate_grid(grid)
    neighbor_directions(connectivity)

    if minimum_cell_count < 1:
        raise ValueError("minimum_cell_count must be at least 1.")

    height = len(validated_grid)
    width = len(validated_grid[0])

    included = set(include_colors) if include_colors is not None else None
    excluded = set(exclude_colors) if exclude_colors is not None else set()

    visited: set[Cell] = set()
    raw_components: list[tuple[int, tuple[Cell, ...]]] = []

    for start_row in range(height):
        for start_column in range(width):
            start = (start_row, start_column)

            if start in visited:
                continue

            color = validated_grid[start_row][start_column]

            if included is not None and color not in included:
                visited.add(start)
                continue

            if color in excluded:
                visited.add(start)
                continue

            queue: deque[Cell] = deque([start])
            visited.add(start)
            component_cells: list[Cell] = []

            while queue:
                row, column = queue.popleft()
                component_cells.append((row, column))

                for next_row, next_column in iter_neighbors(
                    row=row,
                    column=column,
                    height=height,
                    width=width,
                    connectivity=connectivity,
                ):
                    next_cell = (next_row, next_column)

                    if next_cell in visited:
                        continue

                    if validated_grid[next_row][next_column] != color:
                        continue

                    visited.add(next_cell)
                    queue.append(next_cell)

            if len(component_cells) >= minimum_cell_count:
                raw_components.append(
                    (
                        color,
                        tuple(sorted(component_cells)),
                    )
                )

    raw_components.sort(
        key=lambda item: (
            item[1][0][0],
            item[1][0][1],
            item[0],
            -len(item[1]),
        )
    )

    objects = tuple(
        build_visual_object(
            object_id=index,
            color=color,
            cells=cells,
            grid_height=height,
            grid_width=width,
            connectivity=connectivity,
        )
        for index, (color, cells) in enumerate(raw_components, start=1)
    )

    return ObjectDetectionResult(
        height=height,
        width=width,
        connectivity=connectivity,
        object_count=len(objects),
        objects=objects,
    )


def detect_objects_both_connectivities(
    grid: Any,
    include_colors: Iterable[int] | None = None,
    exclude_colors: Iterable[int] | None = None,
    minimum_cell_count: int = 1,
) -> dict[str, ObjectDetectionResult]:
    """
    Run the same grid through both 4-connectivity and 8-connectivity.

    This is useful for visual verification when diagonal contact may or may not
    belong to the same object.
    """
    return {
        "four_connected": detect_objects(
            grid=grid,
            connectivity=4,
            include_colors=include_colors,
            exclude_colors=exclude_colors,
            minimum_cell_count=minimum_cell_count,
        ),
        "eight_connected": detect_objects(
            grid=grid,
            connectivity=8,
            include_colors=include_colors,
            exclude_colors=exclude_colors,
            minimum_cell_count=minimum_cell_count,
        ),
    }


# =============================================================================
# LOOKUP HELPERS
# =============================================================================

def object_by_id(
    result: ObjectDetectionResult,
    object_id: int,
) -> VisualObject | None:
    for visual_object in result.objects:
        if visual_object.object_id == object_id:
            return visual_object

    return None


def objects_by_color(
    result: ObjectDetectionResult,
    color: int,
) -> tuple[VisualObject, ...]:
    return tuple(
        visual_object
        for visual_object in result.objects
        if visual_object.color == color
    )


def cell_to_object_map(
    result: ObjectDetectionResult,
) -> dict[Cell, int]:
    mapping: dict[Cell, int] = {}

    for visual_object in result.objects:
        for cell in visual_object.cells:
            mapping[cell] = visual_object.object_id

    return mapping


# =============================================================================
# DEBUG PRINTING
# =============================================================================

def print_detection_result(result: ObjectDetectionResult) -> None:
    print()
    print(
        f"OBJECT DETECTION "
        f"({result.connectivity}-CONNECTED)"
    )
    print("=" * 72)
    print(
        f"Grid: {result.height}x{result.width} | "
        f"Objects: {result.object_count}"
    )
    print()

    for visual_object in result.objects:
        bbox = visual_object.bbox

        print(
            f"Object {visual_object.object_id:>3} | "
            f"color={visual_object.color} | "
            f"cells={visual_object.cell_count:<4} | "
            f"bbox=(top={bbox.top}, left={bbox.left}, "
            f"h={bbox.height}, w={bbox.width}) | "
            f"border={'yes' if visual_object.touches_border else 'no'}"
        )


# =============================================================================
# SMALL SELF-TEST
# =============================================================================

def _self_test() -> None:
    test_grid = [
        [0, 0, 0, 0, 0],
        [0, 2, 2, 0, 3],
        [0, 2, 0, 0, 3],
        [0, 0, 0, 3, 0],
        [4, 4, 0, 0, 0],
    ]

    print("Including every color:")
    result_all = detect_objects(
        test_grid,
        connectivity=4,
    )
    print_detection_result(result_all)

    print()
    print("Excluding color 0:")
    result_without_zero = detect_objects(
        test_grid,
        connectivity=4,
        exclude_colors={0},
    )
    print_detection_result(result_without_zero)

    print()
    print("Comparing 4-connected and 8-connected:")
    both = detect_objects_both_connectivities(
        test_grid,
        exclude_colors={0},
    )

    print_detection_result(both["four_connected"])
    print_detection_result(both["eight_connected"])


if __name__ == "__main__":
    _self_test()
