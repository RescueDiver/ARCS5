from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Iterable, Sequence

from vision.objects import ObjectDetectionResult, VisualObject


# =============================================================================
# DATA TYPES
# =============================================================================

Cell = tuple[int, int]


@dataclass(frozen=True)
class ObjectRelationship:
    """
    Factual relationship from source_object_id to target_object_id.

    Direction matters:
        source left_of target
        source inside target
        source nearest target
    """

    source_object_id: int
    target_object_id: int

    left_of: bool
    right_of: bool
    above: bool
    below: bool

    overlaps: bool

    edge_touching: bool
    corner_touching: bool
    touching: bool

    inside: bool
    contains: bool

    same_color: bool
    same_cell_count: bool
    same_dimensions: bool
    same_shape: bool

    aligned_row: bool
    aligned_column: bool

    row_gap: int
    column_gap: int

    minimum_manhattan_distance: int
    minimum_chebyshev_distance: int
    center_distance: float

    nearest: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def active_names(self) -> tuple[str, ...]:
        names: list[str] = []

        for name in (
            "left_of",
            "right_of",
            "above",
            "below",
            "overlaps",
            "edge_touching",
            "corner_touching",
            "touching",
            "inside",
            "contains",
            "same_color",
            "same_cell_count",
            "same_dimensions",
            "same_shape",
            "aligned_row",
            "aligned_column",
            "nearest",
        ):
            if getattr(self, name):
                names.append(name)

        return tuple(names)


@dataclass(frozen=True)
class RelationshipResult:
    connectivity: int
    object_count: int
    relationship_count: int
    relationships: tuple[ObjectRelationship, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "connectivity": self.connectivity,
            "object_count": self.object_count,
            "relationship_count": self.relationship_count,
            "relationships": [
                relationship.to_dict()
                for relationship in self.relationships
            ],
        }

    def from_object(
        self,
        object_id: int,
    ) -> tuple[ObjectRelationship, ...]:
        return tuple(
            relationship
            for relationship in self.relationships
            if relationship.source_object_id == object_id
        )

    def between(
        self,
        source_object_id: int,
        target_object_id: int,
    ) -> ObjectRelationship | None:
        for relationship in self.relationships:
            if (
                relationship.source_object_id == source_object_id
                and relationship.target_object_id == target_object_id
            ):
                return relationship

        return None


# =============================================================================
# SMALL GEOMETRY HELPERS
# =============================================================================

def object_center(visual_object: VisualObject) -> tuple[float, float]:
    return (
        (visual_object.bbox.top + visual_object.bbox.bottom) / 2.0,
        (visual_object.bbox.left + visual_object.bbox.right) / 2.0,
    )


def row_gap(
    source: VisualObject,
    target: VisualObject,
) -> int:
    if source.bbox.bottom < target.bbox.top:
        return target.bbox.top - source.bbox.bottom - 1

    if target.bbox.bottom < source.bbox.top:
        return source.bbox.top - target.bbox.bottom - 1

    return 0


def column_gap(
    source: VisualObject,
    target: VisualObject,
) -> int:
    if source.bbox.right < target.bbox.left:
        return target.bbox.left - source.bbox.right - 1

    if target.bbox.right < source.bbox.left:
        return source.bbox.left - target.bbox.right - 1

    return 0


def minimum_manhattan_distance(
    source_cells: Sequence[Cell],
    target_cells: Sequence[Cell],
) -> int:
    return min(
        abs(source_row - target_row)
        + abs(source_column - target_column)
        for source_row, source_column in source_cells
        for target_row, target_column in target_cells
    )


def minimum_chebyshev_distance(
    source_cells: Sequence[Cell],
    target_cells: Sequence[Cell],
) -> int:
    return min(
        max(
            abs(source_row - target_row),
            abs(source_column - target_column),
        )
        for source_row, source_column in source_cells
        for target_row, target_column in target_cells
    )


def translated_shape(visual_object: VisualObject) -> frozenset[Cell]:
    top = visual_object.bbox.top
    left = visual_object.bbox.left

    return frozenset(
        (row - top, column - left)
        for row, column in visual_object.cells
    )


def same_shape(
    source: VisualObject,
    target: VisualObject,
) -> bool:
    return (
        source.height == target.height
        and source.width == target.width
        and translated_shape(source) == translated_shape(target)
    )


def bbox_row_overlap(
    source: VisualObject,
    target: VisualObject,
) -> bool:
    return not (
        source.bbox.bottom < target.bbox.top
        or target.bbox.bottom < source.bbox.top
    )


def bbox_column_overlap(
    source: VisualObject,
    target: VisualObject,
) -> bool:
    return not (
        source.bbox.right < target.bbox.left
        or target.bbox.right < source.bbox.left
    )


# =============================================================================
# TOUCHING AND OVERLAP
# =============================================================================

def overlapping(
    source: VisualObject,
    target: VisualObject,
) -> bool:
    return bool(set(source.cells) & set(target.cells))


def edge_touching(
    source: VisualObject,
    target: VisualObject,
) -> bool:
    target_cells = set(target.cells)

    for row, column in source.cells:
        for row_change, column_change in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            if (row + row_change, column + column_change) in target_cells:
                return True

    return False


def corner_touching(
    source: VisualObject,
    target: VisualObject,
) -> bool:
    target_cells = set(target.cells)

    for row, column in source.cells:
        for row_change, column_change in (
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ):
            if (row + row_change, column + column_change) in target_cells:
                return True

    return False


# =============================================================================
# ENCLOSURE
# =============================================================================

def is_inside(
    inner: VisualObject,
    outer: VisualObject,
) -> bool:
    """
    Return True only when the outer object's cells form a closed barrier
    around every cell of the inner object.

    This is stricter and more useful than simply checking whether the inner
    object's bounding box sits inside the outer object's bounding box.
    """
    if inner.object_id == outer.object_id:
        return False

    outer_cells = set(outer.cells)
    inner_cells = set(inner.cells)

    if outer_cells & inner_cells:
        return False

    top = outer.bbox.top - 1
    bottom = outer.bbox.bottom + 1
    left = outer.bbox.left - 1
    right = outer.bbox.right + 1

    if not all(
        top < row < bottom and left < column < right
        for row, column in inner.cells
    ):
        return False

    start = (top, left)
    queue = [start]
    reachable = {start}

    while queue:
        row, column = queue.pop()

        for row_change, column_change in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            next_cell = (
                row + row_change,
                column + column_change,
            )

            next_row, next_column = next_cell

            if not (
                top <= next_row <= bottom
                and left <= next_column <= right
            ):
                continue

            if next_cell in outer_cells:
                continue

            if next_cell in reachable:
                continue

            reachable.add(next_cell)
            queue.append(next_cell)

    return all(cell not in reachable for cell in inner_cells)


# =============================================================================
# PAIR ANALYSIS
# =============================================================================

def build_relationship(
    source: VisualObject,
    target: VisualObject,
    nearest_target_ids: set[int] | None = None,
) -> ObjectRelationship:
    source_center_row, source_center_column = object_center(source)
    target_center_row, target_center_column = object_center(target)

    source_cells = set(source.cells)
    target_cells = set(target.cells)

    does_overlap = bool(source_cells & target_cells)
    does_edge_touch = edge_touching(source, target)
    does_corner_touch = corner_touching(source, target)

    manhattan_distance = minimum_manhattan_distance(
        source.cells,
        target.cells,
    )
    chebyshev_distance = minimum_chebyshev_distance(
        source.cells,
        target.cells,
    )

    source_inside_target = is_inside(source, target)
    source_contains_target = is_inside(target, source)

    return ObjectRelationship(
        source_object_id=source.object_id,
        target_object_id=target.object_id,

        left_of=source.bbox.right < target.bbox.left,
        right_of=source.bbox.left > target.bbox.right,
        above=source.bbox.bottom < target.bbox.top,
        below=source.bbox.top > target.bbox.bottom,

        overlaps=does_overlap,

        edge_touching=does_edge_touch,
        corner_touching=does_corner_touch,
        touching=does_edge_touch or does_corner_touch,

        inside=source_inside_target,
        contains=source_contains_target,

        same_color=source.color == target.color,
        same_cell_count=source.cell_count == target.cell_count,
        same_dimensions=(
            source.height == target.height
            and source.width == target.width
        ),
        same_shape=same_shape(source, target),

        aligned_row=source_center_row == target_center_row,
        aligned_column=source_center_column == target_center_column,

        row_gap=row_gap(source, target),
        column_gap=column_gap(source, target),

        minimum_manhattan_distance=manhattan_distance,
        minimum_chebyshev_distance=chebyshev_distance,
        center_distance=round(
            sqrt(
                (source_center_row - target_center_row) ** 2
                + (source_center_column - target_center_column) ** 2
            ),
            6,
        ),

        nearest=(
            nearest_target_ids is not None
            and target.object_id in nearest_target_ids
        ),
    )


def nearest_object_ids(
    source: VisualObject,
    objects: Sequence[VisualObject],
) -> set[int]:
    distances: dict[int, int] = {}

    for target in objects:
        if target.object_id == source.object_id:
            continue

        distances[target.object_id] = minimum_manhattan_distance(
            source.cells,
            target.cells,
        )

    if not distances:
        return set()

    best_distance = min(distances.values())

    return {
        object_id
        for object_id, distance in distances.items()
        if distance == best_distance
    }


# =============================================================================
# COMPLETE RELATIONSHIP DETECTION
# =============================================================================

def detect_relationships(
    objects_or_result: Sequence[VisualObject] | ObjectDetectionResult,
) -> RelationshipResult:
    if isinstance(objects_or_result, ObjectDetectionResult):
        objects = tuple(objects_or_result.objects)
        connectivity = objects_or_result.connectivity
    else:
        objects = tuple(objects_or_result)
        connectivity = objects[0].connectivity if objects else 4

    relationships: list[ObjectRelationship] = []

    for source in objects:
        nearest_ids = nearest_object_ids(source, objects)

        for target in objects:
            if source.object_id == target.object_id:
                continue

            relationships.append(
                build_relationship(
                    source=source,
                    target=target,
                    nearest_target_ids=nearest_ids,
                )
            )

    relationships.sort(
        key=lambda relationship: (
            relationship.source_object_id,
            relationship.target_object_id,
        )
    )

    return RelationshipResult(
        connectivity=connectivity,
        object_count=len(objects),
        relationship_count=len(relationships),
        relationships=tuple(relationships),
    )


# =============================================================================
# REPORT HELPERS
# =============================================================================

def relationships_for_object(
    result: RelationshipResult,
    object_id: int,
) -> tuple[ObjectRelationship, ...]:
    return result.from_object(object_id)


def related_object_ids(
    result: RelationshipResult,
    object_id: int,
    relationship_name: str,
) -> tuple[int, ...]:
    if not hasattr(ObjectRelationship, relationship_name):
        valid_names = [
            field_name
            for field_name in ObjectRelationship.__dataclass_fields__
            if field_name not in {
                "source_object_id",
                "target_object_id",
                "row_gap",
                "column_gap",
                "minimum_manhattan_distance",
                "minimum_chebyshev_distance",
                "center_distance",
            }
        ]

        raise ValueError(
            f"Unknown relationship {relationship_name!r}. "
            f"Valid names: {', '.join(valid_names)}"
        )

    return tuple(
        relationship.target_object_id
        for relationship in result.from_object(object_id)
        if bool(getattr(relationship, relationship_name))
    )


def summarize_object_relationships(
    result: RelationshipResult,
    object_id: int,
) -> dict[str, tuple[int, ...]]:
    summary: dict[str, list[int]] = {}

    for relationship in result.from_object(object_id):
        for name in relationship.active_names():
            summary.setdefault(name, []).append(
                relationship.target_object_id
            )

    return {
        name: tuple(sorted(object_ids))
        for name, object_ids in sorted(summary.items())
    }


def format_object_relationships(
    result: RelationshipResult,
    object_id: int,
) -> str:
    summary = summarize_object_relationships(result, object_id)

    lines = [f"Object {object_id} relationships"]

    if not summary:
        lines.append("  none")
        return "\n".join(lines)

    for name, object_ids in summary.items():
        ids_text = ", ".join(str(value) for value in object_ids)
        lines.append(f"  {name}: {ids_text}")

    return "\n".join(lines)


# =============================================================================
# SELF TEST
# =============================================================================

def _self_test() -> None:
    from vision.objects import detect_objects

    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 2, 2, 2, 0],
        [0, 2, 0, 0, 0, 0, 0, 2, 0],
        [0, 2, 0, 3, 3, 0, 0, 2, 0],
        [0, 2, 0, 3, 3, 0, 0, 2, 0],
        [0, 2, 0, 0, 0, 0, 0, 2, 0],
        [0, 2, 2, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 4],
    ]

    detection = detect_objects(
        grid,
        connectivity=4,
        exclude_colors={0},
    )

    relationships = detect_relationships(detection)

    print("=" * 72)
    print("RELATIONSHIPS SELF TEST")
    print("=" * 72)
    print(f"Objects: {relationships.object_count}")
    print(f"Directed relationships: {relationships.relationship_count}")
    print()

    for visual_object in detection.objects:
        print(format_object_relationships(
            relationships,
            visual_object.object_id,
        ))
        print()


if __name__ == "__main__":
    _self_test()
