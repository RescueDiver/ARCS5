from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from vision.pair_story import PairStory, StoryStatement, describe_grid_pair


# =============================================================================
# GENERALIZED STORY DATA TYPES
# =============================================================================

@dataclass(frozen=True)
class GeneralizedEvent:
    """
    One abstract transformation event.

    The event deliberately avoids concrete ARC colors, exact object IDs, and
    pair-specific wording unless those facts are structurally important.
    """

    event_type: str
    subject_type: str
    action: str
    object_type: str | None = None
    properties: tuple[tuple[str, Any], ...] = ()
    source_statement: str = ""

    def property_dict(self) -> dict[str, Any]:
        return dict(self.properties)

    def signature(
        self,
        *,
        include_values: bool = False,
    ) -> tuple[Any, ...]:
        if include_values:
            normalized_properties = self.properties
        else:
            normalized_properties = tuple(
                (name, _generalize_property_value(name, value))
                for name, value in self.properties
            )

        return (
            self.event_type,
            self.subject_type,
            self.action,
            self.object_type,
            normalized_properties,
        )


@dataclass(frozen=True)
class GeneralizedPairStory:
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    events: tuple[GeneralizedEvent, ...]

    @property
    def event_types(self) -> tuple[str, ...]:
        return tuple(event.event_type for event in self.events)

    def signatures(
        self,
        *,
        include_values: bool = False,
    ) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            event.signature(include_values=include_values)
            for event in self.events
        )


@dataclass(frozen=True)
class GeneralizedCommonStory:
    pair_count: int
    shared_event_signatures: tuple[tuple[Any, ...], ...]
    shared_event_types: tuple[str, ...]
    pair_event_counts: tuple[int, ...]


# =============================================================================
# PROPERTY GENERALIZATION
# =============================================================================

def _generalize_number(value: int | float) -> str:
    if value == 0:
        return "zero"

    if value == 1:
        return "one"

    if value == -1:
        return "negative_one"

    if value > 0:
        return "positive"

    return "negative"


def _generalize_property_value(
    name: str,
    value: Any,
) -> Any:
    """
    Replace pair-specific values with reusable categories.

    Exact values remain available in the event itself. This function is used
    only when comparing events across train pairs.
    """

    if name in {
        "row_shift",
        "column_shift",
        "cell_count_delta",
        "height_delta",
        "width_delta",
    }:
        if isinstance(value, (int, float)):
            return _generalize_number(value)

    if name in {
        "input_color",
        "output_color",
        "color",
    }:
        return "color"

    if name in {
        "input_cell_count",
        "output_cell_count",
        "input_height",
        "input_width",
        "output_height",
        "output_width",
    }:
        return "size_value"

    return value


def _properties(
    **values: Any,
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (name, value)
        for name, value in values.items()
        if value is not None
    )


# =============================================================================
# STATEMENT GENERALIZATION
# =============================================================================

def _generalize_movement(
    statement: StoryStatement,
) -> GeneralizedEvent:
    text = statement.text.lower()

    row_direction = "none"
    column_direction = "none"

    if "row" in text and " up" in text:
        row_direction = "up"
    elif "row" in text and " down" in text:
        row_direction = "down"

    if "column" in text and " left" in text:
        column_direction = "left"
    elif "column" in text and " right" in text:
        column_direction = "right"

    return GeneralizedEvent(
        event_type="object_translation",
        subject_type="matched_object",
        action="move",
        properties=_properties(
            movement_kind="translation",
            row_direction=row_direction,
            column_direction=column_direction,
            preserves_identity=True,
        ),
        source_statement=statement.text,
    )


def _generalize_color_change(
    statement: StoryStatement,
) -> GeneralizedEvent:
    return GeneralizedEvent(
        event_type="object_color_change",
        subject_type="matched_object",
        action="recolor",
        properties=_properties(
            preserves_identity=True,
        ),
        source_statement=statement.text,
    )


def _generalize_size_change(
    statement: StoryStatement,
) -> GeneralizedEvent:
    text = statement.text.lower()

    if " grew " in f" {text} ":
        direction = "increase"
    elif " shrank " in f" {text} ":
        direction = "decrease"
    else:
        direction = "changed"

    return GeneralizedEvent(
        event_type="object_size_change",
        subject_type="matched_object",
        action="resize",
        properties=_properties(
            size_direction=direction,
            preserves_identity=True,
        ),
        source_statement=statement.text,
    )


def _generalize_dimension_change(
    statement: StoryStatement,
) -> GeneralizedEvent:
    return GeneralizedEvent(
        event_type="object_dimension_change",
        subject_type="matched_object",
        action="change_dimensions",
        properties=_properties(
            preserves_identity=True,
        ),
        source_statement=statement.text,
    )


def _generalize_shape_change(
    statement: StoryStatement,
) -> GeneralizedEvent:
    return GeneralizedEvent(
        event_type="object_shape_change",
        subject_type="matched_object",
        action="reshape",
        properties=_properties(
            preserves_identity=True,
        ),
        source_statement=statement.text,
    )


def _generalize_role_change(
    statement: StoryStatement,
) -> GeneralizedEvent:
    return GeneralizedEvent(
        event_type="scene_role_change",
        subject_type="matched_object",
        action="change_role",
        properties=_properties(
            preserves_identity=True,
        ),
        source_statement=statement.text,
    )


def _generalize_pattern_change(
    statement: StoryStatement,
) -> GeneralizedEvent:
    return GeneralizedEvent(
        event_type="internal_pattern_change",
        subject_type="matched_object",
        action="change_cell_pattern",
        properties=_properties(
            preserves_identity=True,
        ),
        source_statement=statement.text,
    )


def _generalize_added_object(
    statement: StoryStatement,
) -> GeneralizedEvent:
    return GeneralizedEvent(
        event_type="object_creation",
        subject_type="output_object",
        action="create",
        properties=_properties(
            new_identity=True,
        ),
        source_statement=statement.text,
    )


def _generalize_removed_object(
    statement: StoryStatement,
) -> GeneralizedEvent:
    return GeneralizedEvent(
        event_type="object_deletion",
        subject_type="input_object",
        action="delete",
        properties=_properties(
            identity_removed=True,
        ),
        source_statement=statement.text,
    )


def _generalize_relationship_reversal(
    statement: StoryStatement,
) -> GeneralizedEvent:
    return GeneralizedEvent(
        event_type="relationship_reversal",
        subject_type="matched_object",
        action="reverse_relationship",
        object_type="matched_object",
        properties=_properties(
            directional=True,
        ),
        source_statement=statement.text,
    )


def _generalize_relationship_added(
    statement: StoryStatement,
) -> GeneralizedEvent:
    return GeneralizedEvent(
        event_type="relationship_creation",
        subject_type="matched_object",
        action="add_relationship",
        object_type="matched_object",
        source_statement=statement.text,
    )


def _generalize_relationship_removed(
    statement: StoryStatement,
) -> GeneralizedEvent:
    return GeneralizedEvent(
        event_type="relationship_deletion",
        subject_type="matched_object",
        action="remove_relationship",
        object_type="matched_object",
        source_statement=statement.text,
    )


def _generalize_grid_shape_change(
    statement: StoryStatement,
) -> GeneralizedEvent:
    return GeneralizedEvent(
        event_type="grid_shape_change",
        subject_type="grid",
        action="resize_canvas",
        source_statement=statement.text,
    )


def _generalize_unchanged(
    statement: StoryStatement,
) -> GeneralizedEvent:
    return GeneralizedEvent(
        event_type="object_preservation",
        subject_type="matched_object",
        action="preserve",
        properties=_properties(
            position_preserved=True,
            color_preserved=True,
            size_preserved=True,
            shape_preserved=True,
        ),
        source_statement=statement.text,
    )


STATEMENT_GENERALIZERS = {
    "movement": _generalize_movement,
    "color_change": _generalize_color_change,
    "size_change": _generalize_size_change,
    "dimension_change": _generalize_dimension_change,
    "shape_change": _generalize_shape_change,
    "role_change": _generalize_role_change,
    "pattern_change": _generalize_pattern_change,
    "added_object": _generalize_added_object,
    "removed_object": _generalize_removed_object,
    "relationship_reversal": _generalize_relationship_reversal,
    "relationship_added": _generalize_relationship_added,
    "relationship_removed": _generalize_relationship_removed,
    "grid_shape_change": _generalize_grid_shape_change,
    "unchanged": _generalize_unchanged,
}


def generalize_statement(
    statement: StoryStatement,
) -> GeneralizedEvent | None:
    generalizer = STATEMENT_GENERALIZERS.get(statement.category)

    if generalizer is None:
        return None

    return generalizer(statement)


# =============================================================================
# PAIR STORY GENERALIZATION
# =============================================================================

def _deduplicate_events(
    events: Iterable[GeneralizedEvent],
) -> tuple[GeneralizedEvent, ...]:
    """
    Remove inverse duplicates that describe the same high-level event.

    Example:
        red changed left_of green -> right_of green
        green changed right_of red -> left_of red

    Both reduce to one relationship_reversal event.
    """

    unique: list[GeneralizedEvent] = []
    seen: set[tuple[Any, ...]] = set()

    for event in events:
        signature = event.signature(include_values=False)

        if signature in seen:
            continue

        seen.add(signature)
        unique.append(event)

    return tuple(unique)


def generalize_pair_story(
    story: PairStory,
    *,
    include_preservations: bool = True,
) -> GeneralizedPairStory:
    events: list[GeneralizedEvent] = []

    for statement in story.statements:
        if (
            statement.category == "unchanged"
            and not include_preservations
        ):
            continue

        event = generalize_statement(statement)

        if event is not None:
            events.append(event)

    return GeneralizedPairStory(
        input_shape=story.input_shape,
        output_shape=story.output_shape,
        events=_deduplicate_events(events),
    )


def generalize_grid_pair(
    input_grid: Sequence[Sequence[int]],
    output_grid: Sequence[Sequence[int]],
    *,
    connectivity: int = 4,
    preferred_void_colors: Iterable[int] = (0,),
    background_hint: int | None = None,
    minimum_match_score: float = 0.45,
    include_unchanged: bool = True,
    include_relationship_changes: bool = True,
    include_preservations: bool = True,
) -> GeneralizedPairStory:
    story = describe_grid_pair(
        input_grid,
        output_grid,
        connectivity=connectivity,
        preferred_void_colors=preferred_void_colors,
        background_hint=background_hint,
        minimum_match_score=minimum_match_score,
        include_unchanged=include_unchanged,
        include_relationship_changes=include_relationship_changes,
    )

    return generalize_pair_story(
        story,
        include_preservations=include_preservations,
    )


# =============================================================================
# CROSS-PAIR COMMON STORY
# =============================================================================

def build_generalized_common_story(
    stories: Sequence[GeneralizedPairStory],
) -> GeneralizedCommonStory:
    if not stories:
        return GeneralizedCommonStory(
            pair_count=0,
            shared_event_signatures=(),
            shared_event_types=(),
            pair_event_counts=(),
        )

    signature_sets = [
        set(story.signatures(include_values=False))
        for story in stories
    ]

    shared_signatures = set.intersection(*signature_sets)

    event_type_sets = [
        set(story.event_types)
        for story in stories
    ]

    shared_event_types = set.intersection(*event_type_sets)

    return GeneralizedCommonStory(
        pair_count=len(stories),
        shared_event_signatures=tuple(
            sorted(shared_signatures, key=repr)
        ),
        shared_event_types=tuple(
            sorted(shared_event_types)
        ),
        pair_event_counts=tuple(
            len(story.events)
            for story in stories
        ),
    )


# =============================================================================
# FORMATTING
# =============================================================================

def _format_properties(
    properties: tuple[tuple[str, Any], ...],
) -> str:
    if not properties:
        return ""

    return ", ".join(
        f"{name}={value}"
        for name, value in properties
    )


def format_generalized_pair_story(
    story: GeneralizedPairStory,
) -> str:
    lines = [
        "=" * 72,
        "GENERALIZED PAIR STORY",
        "=" * 72,
        (
            f"Grid: {story.input_shape[0]}x{story.input_shape[1]}"
            f" -> "
            f"{story.output_shape[0]}x{story.output_shape[1]}"
        ),
        f"Events: {len(story.events)}",
        "",
    ]

    if not story.events:
        lines.append("No generalized events were produced.")
        return "\n".join(lines)

    for index, event in enumerate(story.events, start=1):
        target = (
            event.object_type
            if event.object_type is not None
            else "scene"
        )

        lines.append(
            f"{index}. {event.event_type}"
        )
        lines.append(
            f"   {event.subject_type} --{event.action}--> {target}"
        )

        property_text = _format_properties(event.properties)

        if property_text:
            lines.append(f"   properties: {property_text}")

        if event.source_statement:
            lines.append(
                f"   source: {event.source_statement}"
            )

    return "\n".join(lines)


def format_generalized_common_story(
    common: GeneralizedCommonStory,
) -> str:
    lines = [
        "=" * 72,
        "GENERALIZED COMMON STORY",
        "=" * 72,
        f"Train pairs: {common.pair_count}",
        f"Events per pair: {common.pair_event_counts}",
        "",
    ]

    if not common.shared_event_types:
        lines.append(
            "No generalized event type appears in every train pair."
        )
        return "\n".join(lines)

    lines.append("Event types present in every train pair:")

    for event_type in common.shared_event_types:
        lines.append(f"  ✓ {event_type}")

    lines.append("")

    if common.shared_event_signatures:
        lines.append("Fully shared generalized event signatures:")

        for signature in common.shared_event_signatures:
            lines.append(f"  ✓ {signature}")
    else:
        lines.append(
            "No complete event signature is shared by every pair."
        )

    lines.extend([
        "",
        "This is still factual generalization, not a learned executable rule.",
    ])

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

    generalized = generalize_grid_pair(
        input_grid,
        output_grid,
        include_preservations=True,
    )

    print(format_generalized_pair_story(generalized))

    common = build_generalized_common_story([generalized])
    print()
    print(format_generalized_common_story(common))


if __name__ == "__main__":
    _self_test()