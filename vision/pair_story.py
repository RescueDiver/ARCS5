from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from vision.scene_difference import (
    ObjectChange,
    RelationshipChange,
    SceneDifference,
    compare_grids,
)
from vision.scene_graph import (
    SceneGraph,
    SceneGraphNode,
    build_scene_graph,
)


@dataclass(frozen=True)
class StoryStatement:
    category: str
    text: str
    input_object_ids: tuple[int, ...] = ()
    output_object_ids: tuple[int, ...] = ()
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairStory:
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    statements: tuple[StoryStatement, ...]

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(statement.text for statement in self.statements)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "light blue", 9: "maroon",
}

INVERSE_RELATIONSHIPS = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "inside": "contains",
    "contains": "inside",
}

RELATIONSHIP_WORDING = {
    "left_of": "left of",
    "right_of": "right of",
    "above": "above",
    "below": "below",
    "inside": "inside",
    "contains": "containing",
    "touching": "touching",
    "edge_touching": "edge-touching",
    "corner_touching": "corner-touching",
    "overlaps": "overlapping",
    "same_color": "the same color as",
    "same_shape": "the same shape as",
    "same_dimensions": "the same dimensions as",
    "same_cell_count": "the same size as",
    "aligned_row": "row-aligned with",
    "aligned_column": "column-aligned with",
    "nearest": "nearest to",
}


def _color_name(color: int) -> str:
    return COLOR_NAMES.get(color, f"color {color}")


def _node_description(
    node: SceneGraphNode,
    *,
    include_role: bool = False,
) -> str:
    color = _color_name(node.color)

    if node.cell_count == 1:
        shape = "single cell"
    elif node.shape_name == "solid_square":
        shape = f"{node.bbox_height}x{node.bbox_width} square"
    elif node.shape_name == "solid_rectangle":
        shape = f"{node.bbox_height}x{node.bbox_width} rectangle"
    elif node.shape_name == "horizontal_line":
        shape = f"horizontal line of length {node.bbox_width}"
    elif node.shape_name == "vertical_line":
        shape = f"vertical line of length {node.bbox_height}"
    else:
        shape = f"{node.cell_count}-cell object"

    if include_role and node.role.value not in {"foreground", "uncertain"}:
        return f"the {color} {node.role.value} {shape}"

    return f"the {color} {shape}"


def _movement_phrase(row_shift: int, column_shift: int) -> str:
    parts: list[str] = []

    if row_shift < 0:
        amount = abs(row_shift)
        parts.append(f"{amount} row{'s' if amount != 1 else ''} up")
    elif row_shift > 0:
        amount = row_shift
        parts.append(f"{amount} row{'s' if amount != 1 else ''} down")

    if column_shift < 0:
        amount = abs(column_shift)
        parts.append(f"{amount} column{'s' if amount != 1 else ''} left")
    elif column_shift > 0:
        amount = column_shift
        parts.append(f"{amount} column{'s' if amount != 1 else ''} right")

    return " and ".join(parts) if parts else "did not move"


def _relationship_phrase(name: str) -> str:
    return RELATIONSHIP_WORDING.get(name, name.replace("_", " "))


def _is_background_only_noise(
    change: ObjectChange,
    input_graph: SceneGraph,
    output_graph: SceneGraph,
) -> bool:
    input_node = input_graph.node(change.input_object_id)
    output_node = output_graph.node(change.output_object_id)

    if input_node is None or output_node is None:
        return False

    if (
        input_node.role.value != "background"
        or output_node.role.value != "background"
    ):
        return False

    return (
        change.cells_changed
        and not change.moved
        and not change.color_changed
        and not change.cell_count_changed
        and not change.dimensions_changed
        and not change.shape_changed
        and not change.role_changed
    )


def _meaningful_object_changes(
    difference: SceneDifference,
    input_graph: SceneGraph,
    output_graph: SceneGraph,
) -> tuple[ObjectChange, ...]:
    return tuple(
        change
        for change in difference.object_changes
        if not _is_background_only_noise(change, input_graph, output_graph)
    )


def _object_change_statements(
    change: ObjectChange,
    input_graph: SceneGraph,
    output_graph: SceneGraph,
) -> list[StoryStatement]:
    input_node = input_graph.node(change.input_object_id)
    output_node = output_graph.node(change.output_object_id)

    if input_node is None or output_node is None:
        return []

    subject = _node_description(input_node)
    statements: list[StoryStatement] = []

    if change.moved:
        statements.append(
            StoryStatement(
                category="movement",
                text=(
                    f"{subject.capitalize()} moved "
                    f"{_movement_phrase(change.row_shift, change.column_shift)}."
                ),
                input_object_ids=(input_node.object_id,),
                output_object_ids=(output_node.object_id,),
                properties={
                    "row_shift": change.row_shift,
                    "column_shift": change.column_shift,
                    "translation_vector": (
                        change.row_shift,
                        change.column_shift,
                    ),
                    "distance": abs(change.row_shift) + abs(change.column_shift),
                },
            )
        )

    if change.color_changed:
        statements.append(
            StoryStatement(
                category="color_change",
                text=(
                    f"{subject.capitalize()} changed color from "
                    f"{_color_name(change.input_color)} to "
                    f"{_color_name(change.output_color)}."
                ),
                input_object_ids=(input_node.object_id,),
                output_object_ids=(output_node.object_id,),
                properties={
                    "input_color": change.input_color,
                    "output_color": change.output_color,
                    "from_color": change.input_color,
                    "to_color": change.output_color,
                },
            )
        )

    if change.cell_count_changed:
        direction = (
            "grew"
            if change.output_cell_count > change.input_cell_count
            else "shrank"
        )
        statements.append(
            StoryStatement(
                category="size_change",
                text=(
                    f"{subject.capitalize()} {direction} from "
                    f"{change.input_cell_count} cells to "
                    f"{change.output_cell_count} cells."
                ),
                input_object_ids=(input_node.object_id,),
                output_object_ids=(output_node.object_id,),
                properties={
                    "input_cell_count": change.input_cell_count,
                    "output_cell_count": change.output_cell_count,
                    "cell_count_delta": (
                        change.output_cell_count - change.input_cell_count
                    ),
                    "size_direction": (
                        "increase"
                        if change.output_cell_count > change.input_cell_count
                        else "decrease"
                    ),
                },
            )
        )

    if change.dimensions_changed:
        input_height, input_width = change.input_dimensions
        output_height, output_width = change.output_dimensions
        statements.append(
            StoryStatement(
                category="dimension_change",
                text=(
                    f"{subject.capitalize()} changed dimensions from "
                    f"{input_height}x{input_width} to "
                    f"{output_height}x{output_width}."
                ),
                input_object_ids=(input_node.object_id,),
                output_object_ids=(output_node.object_id,),
                properties={
                    "input_dimensions": change.input_dimensions,
                    "output_dimensions": change.output_dimensions,
                    "input_height": input_height,
                    "input_width": input_width,
                    "output_height": output_height,
                    "output_width": output_width,
                    "height_delta": output_height - input_height,
                    "width_delta": output_width - input_width,
                },
            )
        )

    if change.shape_changed:
        statements.append(
            StoryStatement(
                category="shape_change",
                text=(
                    f"{subject.capitalize()} changed shape from "
                    f"{change.input_shape.replace('_', ' ')} to "
                    f"{change.output_shape.replace('_', ' ')}."
                ),
                input_object_ids=(input_node.object_id,),
                output_object_ids=(output_node.object_id,),
                properties={
                    "input_shape": change.input_shape,
                    "output_shape": change.output_shape,
                },
            )
        )

    if change.role_changed:
        statements.append(
            StoryStatement(
                category="role_change",
                text=(
                    f"{subject.capitalize()} changed scene role from "
                    f"{change.input_role} to {change.output_role}."
                ),
                input_object_ids=(input_node.object_id,),
                output_object_ids=(output_node.object_id,),
                properties={
                    "input_role": change.input_role,
                    "output_role": change.output_role,
                },
            )
        )

    changed = any((
        change.moved,
        change.color_changed,
        change.cell_count_changed,
        change.dimensions_changed,
        change.shape_changed,
        change.role_changed,
        change.cells_changed,
    ))

    if not changed:
        statements.append(
            StoryStatement(
                category="unchanged",
                text=f"{subject.capitalize()} remained unchanged.",
                input_object_ids=(input_node.object_id,),
                output_object_ids=(output_node.object_id,),
                properties={
                    "position_preserved": True,
                    "color_preserved": True,
                    "size_preserved": True,
                    "shape_preserved": True,
                },
            )
        )
    elif (
        change.cells_changed
        and not change.moved
        and not change.cell_count_changed
        and not change.dimensions_changed
        and not change.shape_changed
    ):
        statements.append(
            StoryStatement(
                category="pattern_change",
                text=(
                    f"{subject.capitalize()} changed its internal cell pattern."
                ),
                input_object_ids=(input_node.object_id,),
                output_object_ids=(output_node.object_id,),
                properties={"internal_pattern_changed": True},
            )
        )

    return statements


def _object_signature(node: SceneGraphNode) -> tuple[Any, ...]:
    """Intrinsic signature used to group equivalent scene objects."""
    return (
        node.color,
        node.shape_name,
        node.cell_count,
        node.bbox_height,
        node.bbox_width,
        node.role.value,
    )


def _group_nodes(
    graph: SceneGraph,
    object_ids: Sequence[int],
) -> list[list[SceneGraphNode]]:
    groups: dict[tuple[Any, ...], list[SceneGraphNode]] = {}

    for object_id in object_ids:
        node = graph.node(object_id)
        if node is None:
            continue
        groups.setdefault(_object_signature(node), []).append(node)

    ordered = list(groups.values())
    ordered.sort(
        key=lambda group: (
            group[0].role.value == "background",
            group[0].color,
            group[0].cell_count,
            group[0].bbox_height,
            group[0].bbox_width,
            group[0].object_id,
        )
    )
    return ordered


def _axis_spacing(values: Sequence[int]) -> tuple[int, ...]:
    unique = sorted(set(values))
    return tuple(
        right - left
        for left, right in zip(unique, unique[1:])
    )


def _regular_spacing_summary(
    nodes: Sequence[SceneGraphNode],
) -> Mapping[str, Any]:
    """
    Describe whether equivalent objects occupy a regular row/column layout.

    This records an observable collection fact only. It does not claim that
    the collection is the transformation rule.
    """
    if len(nodes) < 3:
        return {
            "regular_layout": False,
            "layout_kind": "collection",
        }

    rows = [node.bbox_top for node in nodes]
    columns = [node.bbox_left for node in nodes]
    unique_rows = sorted(set(rows))
    unique_columns = sorted(set(columns))

    row_steps = _axis_spacing(rows)
    column_steps = _axis_spacing(columns)

    regular_rows = len(row_steps) <= 1 or len(set(row_steps)) == 1
    regular_columns = (
        len(column_steps) <= 1 or len(set(column_steps)) == 1
    )

    row_count = len(unique_rows)
    column_count = len(unique_columns)
    complete_lattice = len(nodes) == row_count * column_count

    if regular_rows and regular_columns and complete_lattice:
        layout_kind = "regular_array"
    elif regular_rows and regular_columns:
        layout_kind = "regular_repetition"
    else:
        layout_kind = "collection"

    return {
        "regular_layout": layout_kind != "collection",
        "layout_kind": layout_kind,
        "row_count": row_count,
        "column_count": column_count,
        "row_spacing": row_steps[0] if row_steps and len(set(row_steps)) == 1 else None,
        "column_spacing": (
            column_steps[0]
            if column_steps and len(set(column_steps)) == 1
            else None
        ),
        "complete_lattice": complete_lattice,
        "top_left_positions": tuple(
            sorted((node.bbox_top, node.bbox_left) for node in nodes)
        ),
    }


def _collection_description(
    node: SceneGraphNode,
    count: int,
) -> str:
    base = _node_description(node)
    if base.startswith("the "):
        base = base[4:]
    return f"{count} identical {base}s"


def _removed_statements(
    difference: SceneDifference,
    input_graph: SceneGraph,
) -> list[StoryStatement]:
    statements: list[StoryStatement] = []

    for nodes in _group_nodes(input_graph, difference.removed_object_ids):
        first = nodes[0]
        object_ids = tuple(node.object_id for node in nodes)

        if len(nodes) == 1:
            text = f"{_node_description(first).capitalize()} was removed."
            category = "removed_object"
        else:
            text = f"{_collection_description(first, len(nodes)).capitalize()} were removed."
            category = "removed_object_group"

        properties = {
            "identity_removed": True,
            "count": len(nodes),
            "color": first.color,
            "cell_count": first.cell_count,
            "dimensions": (first.bbox_height, first.bbox_width),
            "shape": first.shape_name,
            "scene_role": first.role.value,
        }
        properties.update(_regular_spacing_summary(nodes))

        statements.append(
            StoryStatement(
                category=category,
                text=text,
                input_object_ids=object_ids,
                properties=properties,
            )
        )

    return statements


def _added_statements(
    difference: SceneDifference,
    output_graph: SceneGraph,
) -> list[StoryStatement]:
    statements: list[StoryStatement] = []

    for nodes in _group_nodes(output_graph, difference.added_object_ids):
        first = nodes[0]
        object_ids = tuple(node.object_id for node in nodes)
        layout = _regular_spacing_summary(nodes)

        if len(nodes) == 1:
            if first.role.value == "background":
                text = (
                    f"A new {_color_name(first.color)} background canvas "
                    f"was created."
                )
                category = "background_creation"
            else:
                text = f"{_node_description(first).capitalize()} was added."
                category = "added_object"
        elif layout["layout_kind"] == "regular_array":
            text = (
                f"A regular {layout['row_count']}x{layout['column_count']} "
                f"array of {_collection_description(first, len(nodes))} "
                f"was created."
            )
            category = "object_array_creation"
        elif layout["layout_kind"] == "regular_repetition":
            text = (
                f"A regularly spaced repetition of "
                f"{_collection_description(first, len(nodes))} was created."
            )
            category = "object_repetition_creation"
        else:
            text = f"{_collection_description(first, len(nodes)).capitalize()} were added."
            category = "added_object_group"

        properties = {
            "new_identity": True,
            "count": len(nodes),
            "color": first.color,
            "cell_count": first.cell_count,
            "dimensions": (first.bbox_height, first.bbox_width),
            "shape": first.shape_name,
            "scene_role": first.role.value,
        }
        properties.update(layout)

        statements.append(
            StoryStatement(
                category=category,
                text=text,
                output_object_ids=object_ids,
                properties=properties,
            )
        )

    return statements

def _node_is_background(
    graph: SceneGraph,
    object_id: int | None,
) -> bool:
    if object_id is None:
        return False
    node = graph.node(object_id)
    return node is not None and node.role.value == "background"


def _relationship_involves_background(
    change: RelationshipChange,
    input_graph: SceneGraph,
    output_graph: SceneGraph,
) -> bool:
    if change.change_type == "removed":
        graph = input_graph
        source_id = change.input_source_id
        target_id = change.input_target_id
    else:
        graph = output_graph
        source_id = change.output_source_id
        target_id = change.output_target_id

    return (
        _node_is_background(graph, source_id)
        or _node_is_background(graph, target_id)
    )


def _canonical_reversal_key(
    source_id: int,
    target_id: int,
    old_relationship: str,
    new_relationship: str,
) -> tuple[Any, ...]:
    return (
        *tuple(sorted((source_id, target_id))),
        *tuple(sorted((old_relationship, new_relationship))),
    )


def _relationship_statements(
    difference: SceneDifference,
    input_graph: SceneGraph,
    output_graph: SceneGraph,
) -> list[StoryStatement]:
    statements: list[StoryStatement] = []

    removed = [
        change
        for change in difference.relationship_changes
        if (
            change.change_type == "removed"
            and not _relationship_involves_background(
                change,
                input_graph,
                output_graph,
            )
        )
    ]
    added = [
        change
        for change in difference.relationship_changes
        if (
            change.change_type == "added"
            and not _relationship_involves_background(
                change,
                input_graph,
                output_graph,
            )
        )
    ]

    used_added: set[int] = set()
    seen_reversals: set[tuple[Any, ...]] = set()
    seen_statements: set[tuple[Any, ...]] = set()

    for removed_change in removed:
        inverse_name = INVERSE_RELATIONSHIPS.get(
            removed_change.relationship
        )
        paired_index: int | None = None

        if inverse_name is not None:
            for index, added_change in enumerate(added):
                if index in used_added:
                    continue
                if (
                    added_change.relationship == inverse_name
                    and added_change.input_source_id
                    == removed_change.input_source_id
                    and added_change.input_target_id
                    == removed_change.input_target_id
                ):
                    paired_index = index
                    break

        source = input_graph.node(removed_change.input_source_id or -1)
        target = input_graph.node(removed_change.input_target_id or -1)

        if source is None or target is None:
            continue

        source_text = _node_description(source)
        target_text = _node_description(target)

        if paired_index is not None:
            added_change = added[paired_index]
            used_added.add(paired_index)
            reversal_key = _canonical_reversal_key(
                source.object_id,
                target.object_id,
                removed_change.relationship,
                added_change.relationship,
            )

            if reversal_key in seen_reversals:
                continue
            seen_reversals.add(reversal_key)

            statements.append(
                StoryStatement(
                    category="relationship_reversal",
                    text=(
                        f"{source_text.capitalize()} changed from "
                        f"{_relationship_phrase(removed_change.relationship)} "
                        f"{target_text} to "
                        f"{_relationship_phrase(added_change.relationship)} "
                        f"{target_text}."
                    ),
                    input_object_ids=(source.object_id, target.object_id),
                    output_object_ids=tuple(
                        object_id
                        for object_id in (
                            added_change.output_source_id,
                            added_change.output_target_id,
                        )
                        if object_id is not None
                    ),
                    properties={
                        "old_relationship": removed_change.relationship,
                        "new_relationship": added_change.relationship,
                        "directional": True,
                    },
                )
            )
        else:
            statement_key = (
                "removed",
                removed_change.relationship,
                tuple(sorted((source.object_id, target.object_id))),
            )
            if statement_key in seen_statements:
                continue
            seen_statements.add(statement_key)

            statements.append(
                StoryStatement(
                    category="relationship_removed",
                    text=(
                        f"{source_text.capitalize()} was no longer "
                        f"{_relationship_phrase(removed_change.relationship)} "
                        f"{target_text}."
                    ),
                    input_object_ids=(source.object_id, target.object_id),
                    properties={
                        "relationship": removed_change.relationship,
                    },
                )
            )

    for index, added_change in enumerate(added):
        if index in used_added:
            continue

        source = output_graph.node(added_change.output_source_id or -1)
        target = output_graph.node(added_change.output_target_id or -1)

        if source is None or target is None:
            continue

        statement_key = (
            "added",
            added_change.relationship,
            tuple(sorted((source.object_id, target.object_id))),
        )
        if statement_key in seen_statements:
            continue
        seen_statements.add(statement_key)

        statements.append(
            StoryStatement(
                category="relationship_added",
                text=(
                    f"{_node_description(source).capitalize()} became "
                    f"{_relationship_phrase(added_change.relationship)} "
                    f"{_node_description(target)}."
                ),
                output_object_ids=(source.object_id, target.object_id),
                properties={"relationship": added_change.relationship},
            )
        )

    return statements


def build_pair_story(
    difference: SceneDifference,
    input_graph: SceneGraph,
    output_graph: SceneGraph,
    *,
    include_unchanged: bool = True,
    include_relationship_changes: bool = True,
) -> PairStory:
    statements: list[StoryStatement] = []

    if difference.shape_changed:
        statements.append(
            StoryStatement(
                category="grid_shape_change",
                text=(
                    f"The grid changed from "
                    f"{difference.input_shape[0]}x{difference.input_shape[1]} "
                    f"to "
                    f"{difference.output_shape[0]}x{difference.output_shape[1]}."
                ),
                properties={
                    "input_shape": difference.input_shape,
                    "output_shape": difference.output_shape,
                    "height_delta": (
                        difference.output_shape[0] - difference.input_shape[0]
                    ),
                    "width_delta": (
                        difference.output_shape[1] - difference.input_shape[1]
                    ),
                },
            )
        )

    for change in _meaningful_object_changes(
        difference,
        input_graph,
        output_graph,
    ):
        for statement in _object_change_statements(
            change,
            input_graph,
            output_graph,
        ):
            if (
                statement.category == "unchanged"
                and not include_unchanged
            ):
                continue
            statements.append(statement)

    statements.extend(_removed_statements(difference, input_graph))
    statements.extend(_added_statements(difference, output_graph))

    if include_relationship_changes:
        statements.extend(
            _relationship_statements(
                difference,
                input_graph,
                output_graph,
            )
        )

    if not statements:
        statements.append(
            StoryStatement(
                category="no_change",
                text="No meaningful scene change was detected.",
            )
        )

    return PairStory(
        input_shape=difference.input_shape,
        output_shape=difference.output_shape,
        statements=tuple(statements),
    )


def describe_grid_pair(
    input_grid: Sequence[Sequence[int]],
    output_grid: Sequence[Sequence[int]],
    *,
    connectivity: int = 4,
    preferred_void_colors: Iterable[int] = (0,),
    background_hint: int | None = None,
    minimum_match_score: float = 0.65,
    include_unchanged: bool = True,
    include_relationship_changes: bool = True,
) -> PairStory:
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
    difference = compare_grids(
        input_grid,
        output_grid,
        connectivity=connectivity,
        preferred_void_colors=preferred_void_colors,
        background_hint=background_hint,
        minimum_match_score=minimum_match_score,
    )

    return build_pair_story(
        difference,
        input_graph,
        output_graph,
        include_unchanged=include_unchanged,
        include_relationship_changes=include_relationship_changes,
    )


def format_pair_story(story: PairStory) -> str:
    lines = ["=" * 72, "PAIR STORY", "=" * 72]

    for index, statement in enumerate(story.statements, start=1):
        lines.append(f"{index}. {statement.text}")
        if statement.properties:
            lines.append(
                "   properties: "
                + ", ".join(
                    f"{name}={value}"
                    for name, value in sorted(
                        statement.properties.items()
                    )
                )
            )

    return "\n".join(lines)


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

    print(
        format_pair_story(
            describe_grid_pair(
                input_grid,
                output_grid,
                connectivity=4,
            )
        )
    )


if __name__ == "__main__":
    _self_test()