from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Sequence

from vision.scene_graph import (
    SceneGraph,
    SceneGraphEdge,
    SceneGraphNode,
    build_scene_graph,
)


# =============================================================================
# CHANGE TYPES
# =============================================================================

@dataclass(frozen=True)
class ObjectMatch:
    input_object_id: int
    output_object_id: int
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ObjectChange:
    input_object_id: int
    output_object_id: int

    moved: bool
    row_shift: int
    column_shift: int

    color_changed: bool
    input_color: int
    output_color: int

    cell_count_changed: bool
    input_cell_count: int
    output_cell_count: int

    dimensions_changed: bool
    input_dimensions: tuple[int, int]
    output_dimensions: tuple[int, int]

    shape_changed: bool
    input_shape: str
    output_shape: str

    role_changed: bool
    input_role: str
    output_role: str

    cells_changed: bool
    preserved_translation: bool


@dataclass(frozen=True)
class RelationshipChange:
    change_type: str
    relationship: str

    input_source_id: int | None
    input_target_id: int | None

    output_source_id: int | None
    output_target_id: int | None


@dataclass(frozen=True)
class SceneDifference:
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]

    matches: tuple[ObjectMatch, ...]
    object_changes: tuple[ObjectChange, ...]

    removed_object_ids: tuple[int, ...]
    added_object_ids: tuple[int, ...]

    relationship_changes: tuple[RelationshipChange, ...]

    @property
    def shape_changed(self) -> bool:
        return self.input_shape != self.output_shape

    @property
    def unchanged_match_count(self) -> int:
        count = 0

        for change in self.object_changes:
            if not any((
                change.moved,
                change.color_changed,
                change.cell_count_changed,
                change.dimensions_changed,
                change.shape_changed,
                change.role_changed,
                change.cells_changed,
            )):
                count += 1

        return count


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def _normalized_center(
    node: SceneGraphNode,
    graph: SceneGraph,
) -> tuple[float, float]:
    row, column = node.center

    row_denominator = max(1, graph.height - 1)
    column_denominator = max(1, graph.width - 1)

    return (
        row / row_denominator,
        column / column_denominator,
    )


def _center_distance(
    input_node: SceneGraphNode,
    output_node: SceneGraphNode,
    input_graph: SceneGraph,
    output_graph: SceneGraph,
) -> float:
    input_row, input_column = _normalized_center(
        input_node,
        input_graph,
    )
    output_row, output_column = _normalized_center(
        output_node,
        output_graph,
    )

    return hypot(
        input_row - output_row,
        input_column - output_column,
    )


def _translated_cells(
    cells: Iterable[tuple[int, int]],
    row_shift: int,
    column_shift: int,
) -> frozenset[tuple[int, int]]:
    return frozenset(
        (row + row_shift, column + column_shift)
        for row, column in cells
    )


def _preserves_translation(
    input_node: SceneGraphNode,
    output_node: SceneGraphNode,
) -> bool:
    row_shift = output_node.bbox_top - input_node.bbox_top
    column_shift = output_node.bbox_left - input_node.bbox_left

    return _translated_cells(
        input_node.cells,
        row_shift,
        column_shift,
    ) == frozenset(output_node.cells)


# =============================================================================
# MATCHING
# =============================================================================

def _match_score(
    input_node: SceneGraphNode,
    output_node: SceneGraphNode,
    input_graph: SceneGraph,
    output_graph: SceneGraph,
) -> tuple[float, tuple[str, ...]]:
    """
    Score whether two nodes likely represent the same logical object.

    This intentionally favors intrinsic properties over absolute position,
    because movement is a common ARC transformation.
    """
    score = 0.0
    reasons: list[str] = []

    if input_node.color == output_node.color:
        score += 0.22
        reasons.append("same color")

    if input_node.cell_count == output_node.cell_count:
        score += 0.22
        reasons.append("same cell count")
    else:
        larger = max(input_node.cell_count, output_node.cell_count)
        smaller = min(input_node.cell_count, output_node.cell_count)

        if larger > 0 and smaller / larger >= 0.70:
            score += 0.08
            reasons.append("similar cell count")

    if (
        input_node.bbox_height == output_node.bbox_height
        and input_node.bbox_width == output_node.bbox_width
    ):
        score += 0.18
        reasons.append("same dimensions")

    if input_node.shape_name == output_node.shape_name:
        score += 0.18
        reasons.append("same shape")

    if input_node.role == output_node.role:
        score += 0.08
        reasons.append("same scene role")

    if _preserves_translation(input_node, output_node):
        score += 0.22
        reasons.append("same cells under translation")

    distance = _center_distance(
        input_node,
        output_node,
        input_graph,
        output_graph,
    )

    if distance <= 0.10:
        score += 0.10
        reasons.append("very similar normalized position")
    elif distance <= 0.25:
        score += 0.05
        reasons.append("similar normalized position")

    return min(1.0, score), tuple(reasons)


def match_scene_objects(
    input_graph: SceneGraph,
    output_graph: SceneGraph,
    *,
    minimum_score: float = 0.45,
) -> tuple[ObjectMatch, ...]:
    """
    Greedy one-to-one matching using highest-confidence candidate pairs.

    The result is deterministic and conservative. Weak candidates remain
    unmatched and later appear as removed or added objects.
    """
    candidates: list[
        tuple[float, int, int, tuple[str, ...]]
    ] = []

    for input_node in input_graph.nodes:
        for output_node in output_graph.nodes:
            score, reasons = _match_score(
                input_node,
                output_node,
                input_graph,
                output_graph,
            )

            if score >= minimum_score:
                candidates.append((
                    score,
                    input_node.object_id,
                    output_node.object_id,
                    reasons,
                ))

    candidates.sort(
        key=lambda item: (
            item[0],
            -item[1],
            -item[2],
        ),
        reverse=True,
    )

    used_input_ids: set[int] = set()
    used_output_ids: set[int] = set()
    matches: list[ObjectMatch] = []

    for score, input_id, output_id, reasons in candidates:
        if input_id in used_input_ids:
            continue

        if output_id in used_output_ids:
            continue

        used_input_ids.add(input_id)
        used_output_ids.add(output_id)

        matches.append(
            ObjectMatch(
                input_object_id=input_id,
                output_object_id=output_id,
                score=round(score, 3),
                reasons=reasons,
            )
        )

    matches.sort(key=lambda match: match.input_object_id)
    return tuple(matches)


# =============================================================================
# OBJECT CHANGES
# =============================================================================

def _compare_matched_objects(
    input_node: SceneGraphNode,
    output_node: SceneGraphNode,
) -> ObjectChange:
    row_shift = output_node.bbox_top - input_node.bbox_top
    column_shift = output_node.bbox_left - input_node.bbox_left

    translation_preserved = _preserves_translation(
        input_node,
        output_node,
    )

    moved = row_shift != 0 or column_shift != 0

    same_absolute_cells = (
        frozenset(input_node.cells)
        == frozenset(output_node.cells)
    )

    return ObjectChange(
        input_object_id=input_node.object_id,
        output_object_id=output_node.object_id,

        moved=moved,
        row_shift=row_shift,
        column_shift=column_shift,

        color_changed=input_node.color != output_node.color,
        input_color=input_node.color,
        output_color=output_node.color,

        cell_count_changed=(
            input_node.cell_count != output_node.cell_count
        ),
        input_cell_count=input_node.cell_count,
        output_cell_count=output_node.cell_count,

        dimensions_changed=(
            input_node.bbox_height != output_node.bbox_height
            or input_node.bbox_width != output_node.bbox_width
        ),
        input_dimensions=(
            input_node.bbox_height,
            input_node.bbox_width,
        ),
        output_dimensions=(
            output_node.bbox_height,
            output_node.bbox_width,
        ),

        shape_changed=(
            input_node.shape_name != output_node.shape_name
        ),
        input_shape=input_node.shape_name,
        output_shape=output_node.shape_name,

        role_changed=input_node.role != output_node.role,
        input_role=input_node.role.value,
        output_role=output_node.role.value,

        cells_changed=not same_absolute_cells,
        preserved_translation=translation_preserved,
    )


# =============================================================================
# RELATIONSHIP CHANGES
# =============================================================================

def _edge_key(
    edge: SceneGraphEdge,
) -> tuple[int, str, int]:
    return (
        edge.source_id,
        edge.relationship,
        edge.target_id,
    )


def _compare_relationships(
    input_graph: SceneGraph,
    output_graph: SceneGraph,
    matches: tuple[ObjectMatch, ...],
) -> tuple[RelationshipChange, ...]:
    input_to_output = {
        match.input_object_id: match.output_object_id
        for match in matches
    }

    output_edge_keys = {
        _edge_key(edge)
        for edge in output_graph.edges
    }

    input_edge_keys = {
        _edge_key(edge)
        for edge in input_graph.edges
    }

    changes: list[RelationshipChange] = []

    # Relationships removed among matched objects.
    for edge in input_graph.edges:
        mapped_source = input_to_output.get(edge.source_id)
        mapped_target = input_to_output.get(edge.target_id)

        if mapped_source is None or mapped_target is None:
            continue

        mapped_key = (
            mapped_source,
            edge.relationship,
            mapped_target,
        )

        if mapped_key not in output_edge_keys:
            changes.append(
                RelationshipChange(
                    change_type="removed",
                    relationship=edge.relationship,
                    input_source_id=edge.source_id,
                    input_target_id=edge.target_id,
                    output_source_id=mapped_source,
                    output_target_id=mapped_target,
                )
            )

    output_to_input = {
        match.output_object_id: match.input_object_id
        for match in matches
    }

    # Relationships added among matched objects.
    for edge in output_graph.edges:
        mapped_source = output_to_input.get(edge.source_id)
        mapped_target = output_to_input.get(edge.target_id)

        if mapped_source is None or mapped_target is None:
            continue

        mapped_key = (
            mapped_source,
            edge.relationship,
            mapped_target,
        )

        if mapped_key not in input_edge_keys:
            changes.append(
                RelationshipChange(
                    change_type="added",
                    relationship=edge.relationship,
                    input_source_id=mapped_source,
                    input_target_id=mapped_target,
                    output_source_id=edge.source_id,
                    output_target_id=edge.target_id,
                )
            )

    changes.sort(
        key=lambda change: (
            change.change_type,
            change.relationship,
            change.input_source_id or -1,
            change.input_target_id or -1,
        )
    )

    return tuple(changes)


# =============================================================================
# MAIN DIFFERENCE ENGINE
# =============================================================================

def compare_scene_graphs(
    input_graph: SceneGraph,
    output_graph: SceneGraph,
    *,
    minimum_match_score: float = 0.65,
) -> SceneDifference:
    matches = match_scene_objects(
        input_graph,
        output_graph,
        minimum_score=minimum_match_score,
    )

    matched_input_ids = {
        match.input_object_id
        for match in matches
    }

    matched_output_ids = {
        match.output_object_id
        for match in matches
    }

    removed_object_ids = tuple(
        node.object_id
        for node in input_graph.nodes
        if node.object_id not in matched_input_ids
    )

    added_object_ids = tuple(
        node.object_id
        for node in output_graph.nodes
        if node.object_id not in matched_output_ids
    )

    object_changes: list[ObjectChange] = []

    for match in matches:
        input_node = input_graph.node(match.input_object_id)
        output_node = output_graph.node(match.output_object_id)

        if input_node is None or output_node is None:
            continue

        object_changes.append(
            _compare_matched_objects(
                input_node,
                output_node,
            )
        )

    relationship_changes = _compare_relationships(
        input_graph,
        output_graph,
        matches,
    )

    return SceneDifference(
        input_shape=(input_graph.height, input_graph.width),
        output_shape=(output_graph.height, output_graph.width),
        matches=matches,
        object_changes=tuple(object_changes),
        removed_object_ids=removed_object_ids,
        added_object_ids=added_object_ids,
        relationship_changes=relationship_changes,
    )


def compare_grids(
    input_grid: Sequence[Sequence[int]],
    output_grid: Sequence[Sequence[int]],
    *,
    connectivity: int = 4,
    preferred_void_colors: Iterable[int] = (0,),
    background_hint: int | None = None,
    minimum_match_score: float = 0.65,
) -> SceneDifference:
    input_graph = build_scene_graph(
        input_grid,
        connectivity=connectivity,
        preferred_void_colors=preferred_void_colors,
        background_hint=background_hint,
    )

    output_graph = build_scene_graph(
        output_grid,
        connectivity=connectivity,
        preferred_void_colors=preferred_void_colors,
        background_hint=background_hint,
    )

    return compare_scene_graphs(
        input_graph,
        output_graph,
        minimum_match_score=minimum_match_score,
    )


# =============================================================================
# FORMATTING
# =============================================================================

def _change_descriptions(
    change: ObjectChange,
) -> list[str]:
    descriptions: list[str] = []

    if change.moved:
        descriptions.append(
            f"moved by rows={change.row_shift}, "
            f"columns={change.column_shift}"
        )

    if change.color_changed:
        descriptions.append(
            f"color {change.input_color} -> {change.output_color}"
        )

    if change.cell_count_changed:
        descriptions.append(
            f"cells {change.input_cell_count} -> "
            f"{change.output_cell_count}"
        )

    if change.dimensions_changed:
        descriptions.append(
            f"dimensions {change.input_dimensions} -> "
            f"{change.output_dimensions}"
        )

    if change.shape_changed:
        descriptions.append(
            f"shape {change.input_shape} -> {change.output_shape}"
        )

    if change.role_changed:
        descriptions.append(
            f"role {change.input_role} -> {change.output_role}"
        )

    if change.cells_changed and not change.moved:
        descriptions.append("cell pattern changed")

    if change.moved and change.preserved_translation:
        descriptions.append("shape preserved under translation")

    return descriptions


def format_scene_difference(
    difference: SceneDifference,
) -> str:
    lines = [
        "=" * 72,
        "SCENE DIFFERENCE",
        "=" * 72,
        (
            f"Input shape: "
            f"{difference.input_shape[0]}x{difference.input_shape[1]}"
        ),
        (
            f"Output shape: "
            f"{difference.output_shape[0]}x{difference.output_shape[1]}"
        ),
        f"Shape changed: {'yes' if difference.shape_changed else 'no'}",
        "",
        "OBJECT MATCHES",
        "-" * 72,
    ]

    if not difference.matches:
        lines.append("No object matches.")
    else:
        for match in difference.matches:
            lines.append(
                f"Input Object {match.input_object_id} -> "
                f"Output Object {match.output_object_id} "
                f"(score={match.score:.3f})"
            )

            if match.reasons:
                lines.append(
                    "  match reasons: "
                    + ", ".join(match.reasons)
                )

            change = next(
                (
                    item
                    for item in difference.object_changes
                    if item.input_object_id
                    == match.input_object_id
                ),
                None,
            )

            if change is None:
                continue

            descriptions = _change_descriptions(change)

            if descriptions:
                for description in descriptions:
                    lines.append(f"  - {description}")
            else:
                lines.append("  - unchanged")

    lines.extend([
        "",
        "REMOVED OBJECTS",
        "-" * 72,
        (
            ", ".join(
                f"Object {object_id}"
                for object_id in difference.removed_object_ids
            )
            or "none"
        ),
        "",
        "ADDED OBJECTS",
        "-" * 72,
        (
            ", ".join(
                f"Object {object_id}"
                for object_id in difference.added_object_ids
            )
            or "none"
        ),
        "",
        "RELATIONSHIP CHANGES",
        "-" * 72,
    ])

    if not difference.relationship_changes:
        lines.append("none")
    else:
        for change in difference.relationship_changes:
            lines.append(
                f"{change.change_type}: "
                f"Input Object {change.input_source_id} "
                f"{change.relationship} "
                f"Input Object {change.input_target_id}"
            )

    return "\n".join(lines)


# =============================================================================
# SELF TEST
# =============================================================================

def _self_test() -> None:
    input_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 0],
        [0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    output_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 2],
        [0, 0, 0, 0, 2, 2],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    difference = compare_grids(
        input_grid,
        output_grid,
        connectivity=4,
    )

    print(format_scene_difference(difference))


if __name__ == "__main__":
    _self_test()