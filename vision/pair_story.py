from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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


# =============================================================================
# STORY DATA TYPES
# =============================================================================

@dataclass(frozen=True)
class StoryStatement:
    category: str
    text: str
    input_object_ids: tuple[int, ...] = ()
    output_object_ids: tuple[int, ...] = ()


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


# =============================================================================
# WORDING HELPERS
# =============================================================================

COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "magenta",
    7: "orange",
    8: "light blue",
    9: "maroon",
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

    if include_role and node.role.value not in {
        "foreground",
        "uncertain",
    }:
        return f"the {color} {node.role.value} {shape}"

    return f"the {color} {shape}"


def _movement_phrase(
    row_shift: int,
    column_shift: int,
) -> str:
    parts: list[str] = []

    if row_shift < 0:
        amount = abs(row_shift)
        parts.append(
            f"{amount} row{'s' if amount != 1 else ''} up"
        )
    elif row_shift > 0:
        amount = row_shift
        parts.append(
            f"{amount} row{'s' if amount != 1 else ''} down"
        )

    if column_shift < 0:
        amount = abs(column_shift)
        parts.append(
            f"{amount} column{'s' if amount != 1 else ''} left"
        )
    elif column_shift > 0:
        amount = column_shift
        parts.append(
            f"{amount} column{'s' if amount != 1 else ''} right"
        )

    if not parts:
        return "did not move"

    return " and ".join(parts)


def _relationship_phrase(name: str) -> str:
    return RELATIONSHIP_WORDING.get(
        name,
        name.replace("_", " "),
    )


# =============================================================================
# CHANGE FILTERING
# =============================================================================

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
        if not _is_background_only_noise(
            change,
            input_graph,
            output_graph,
        )
    )


# =============================================================================
# OBJECT STORY
# =============================================================================

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
            )
        )

    if change.dimensions_changed:
        statements.append(
            StoryStatement(
                category="dimension_change",
                text=(
                    f"{subject.capitalize()} changed dimensions from "
                    f"{change.input_dimensions[0]}x"
                    f"{change.input_dimensions[1]} to "
                    f"{change.output_dimensions[0]}x"
                    f"{change.output_dimensions[1]}."
                ),
                input_object_ids=(input_node.object_id,),
                output_object_ids=(output_node.object_id,),
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
                    f"{subject.capitalize()} changed its internal "
                    f"cell pattern."
                ),
                input_object_ids=(input_node.object_id,),
                output_object_ids=(output_node.object_id,),
            )
        )

    return statements


# =============================================================================
# ADDED AND REMOVED OBJECTS
# =============================================================================

def _removed_statements(
    difference: SceneDifference,
    input_graph: SceneGraph,
) -> list[StoryStatement]:
    statements: list[StoryStatement] = []

    for object_id in difference.removed_object_ids:
        node = input_graph.node(object_id)

        if node is None:
            continue

        statements.append(
            StoryStatement(
                category="removed_object",
                text=(
                    f"{_node_description(node).capitalize()} "
                    f"was removed."
                ),
                input_object_ids=(object_id,),
            )
        )

    return statements


def _added_statements(
    difference: SceneDifference,
    output_graph: SceneGraph,
) -> list[StoryStatement]:
    statements: list[StoryStatement] = []

    for object_id in difference.added_object_ids:
        node = output_graph.node(object_id)

        if node is None:
            continue

        statements.append(
            StoryStatement(
                category="added_object",
                text=(
                    f"{_node_description(node).capitalize()} "
                    f"was added."
                ),
                output_object_ids=(object_id,),
            )
        )

    return statements


# =============================================================================
# RELATIONSHIP STORY
# =============================================================================

def _relationship_change_key(
    change: RelationshipChange,
) -> tuple[
    str,
    int | None,
    int | None,
    int | None,
    int | None,
]:
    return (
        change.relationship,
        change.input_source_id,
        change.input_target_id,
        change.output_source_id,
        change.output_target_id,
    )


def _node_is_background(
    graph: SceneGraph,
    object_id: int | None,
) -> bool:
    if object_id is None:
        return False

    node = graph.node(object_id)

    return (
        node is not None
        and node.role.value == "background"
    )


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
    object_pair = tuple(sorted((source_id, target_id)))
    relationship_pair = tuple(
        sorted((old_relationship, new_relationship))
    )

    return object_pair + relationship_pair


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

        source = input_graph.node(
            removed_change.input_source_id or -1
        )
        target = input_graph.node(
            removed_change.input_target_id or -1
        )

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
                    input_object_ids=(
                        source.object_id,
                        target.object_id,
                    ),
                    output_object_ids=tuple(
                        object_id
                        for object_id in (
                            added_change.output_source_id,
                            added_change.output_target_id,
                        )
                        if object_id is not None
                    ),
                )
            )
        else:
            statement_key = (
                "removed",
                removed_change.relationship,
                tuple(sorted((
                    source.object_id,
                    target.object_id,
                ))),
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
                    input_object_ids=(
                        source.object_id,
                        target.object_id,
                    ),
                )
            )

    for index, added_change in enumerate(added):
        if index in used_added:
            continue

        source = output_graph.node(
            added_change.output_source_id or -1
        )
        target = output_graph.node(
            added_change.output_target_id or -1
        )

        if source is None or target is None:
            continue

        statement_key = (
            "added",
            added_change.relationship,
            tuple(sorted((
                source.object_id,
                target.object_id,
            ))),
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
                output_object_ids=(
                    source.object_id,
                    target.object_id,
                ),
            )
        )

    return statements


# =============================================================================
# STORY CONSTRUCTION
# =============================================================================

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
                    f"{difference.input_shape[0]}x"
                    f"{difference.input_shape[1]} to "
                    f"{difference.output_shape[0]}x"
                    f"{difference.output_shape[1]}."
                ),
            )
        )

    for change in _meaningful_object_changes(
        difference,
        input_graph,
        output_graph,
    ):
        object_statements = _object_change_statements(
            change,
            input_graph,
            output_graph,
        )

        for statement in object_statements:
            if (
                statement.category == "unchanged"
                and not include_unchanged
            ):
                continue

            statements.append(statement)

    statements.extend(
        _removed_statements(
            difference,
            input_graph,
        )
    )

    statements.extend(
        _added_statements(
            difference,
            output_graph,
        )
    )

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
    minimum_match_score: float = 0.45,
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


# =============================================================================
# FORMATTING
# =============================================================================

def format_pair_story(
    story: PairStory,
) -> str:
    lines = [
        "=" * 72,
        "PAIR STORY",
        "=" * 72,
    ]

    for index, statement in enumerate(
        story.statements,
        start=1,
    ):
        lines.append(f"{index}. {statement.text}")

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

    story = describe_grid_pair(
        input_grid,
        output_grid,
        connectivity=4,
    )

    print(format_pair_story(story))


if __name__ == "__main__":
    _self_test()