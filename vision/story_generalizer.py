from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from vision.pair_story import PairStory, StoryStatement, describe_grid_pair


# =============================================================================
# GENERALIZED STORY DATA TYPES
# =============================================================================

@dataclass(frozen=True)
class GeneralizedEvent:
    """
    One factual, abstract transformation event.

    Important design rule:
    - concrete properties are preserved in ``properties``;
    - generalized properties are used only for cross-pair comparison;
    - unknown future Pair Story categories are not silently discarded.
    """

    event_type: str
    subject_type: str
    action: str
    object_type: str | None = None
    properties: tuple[tuple[str, Any], ...] = ()
    source_category: str = ""
    source_statement: str = ""
    input_object_ids: tuple[int, ...] = ()
    output_object_ids: tuple[int, ...] = ()

    def property_dict(self) -> dict[str, Any]:
        return dict(self.properties)

    def signature(
        self,
        *,
        include_values: bool = False,
        include_source_category: bool = False,
    ) -> tuple[Any, ...]:
        normalized_properties = (
            self.properties
            if include_values
            else tuple(
                (name, _generalize_property_value(name, value))
                for name, value in self.properties
            )
        )

        signature: tuple[Any, ...] = (
            self.event_type,
            self.subject_type,
            self.action,
            self.object_type,
            normalized_properties,
        )

        if include_source_category:
            signature += (self.source_category,)

        return signature


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
    shared_event_type_counts: tuple[tuple[str, int], ...] = ()
    shared_signature_counts: tuple[tuple[tuple[Any, ...], int], ...] = ()


# =============================================================================
# PROPERTY NORMALIZATION
# =============================================================================

COLOR_PROPERTY_NAMES = {
    "color",
    "input_color",
    "output_color",
    "from_color",
    "to_color",
    "target_color",
    "source_color",
    "background_color",
    "foreground_color",
    "divider_color",
    "marker_color",
    "fill_color",
}

COUNT_PROPERTY_NAMES = {
    "count",
    "cell_count",
    "input_cell_count",
    "output_cell_count",
    "row_count",
    "column_count",
    "object_count",
    "repeat_count",
    "layer_count",
    "ring_count",
    "segment_count",
}

DIMENSION_PROPERTY_NAMES = {
    "dimensions",
    "input_dimensions",
    "output_dimensions",
    "object_dimensions",
    "tile_dimensions",
    "unit_dimensions",
    "bbox_dimensions",
}

SHAPE_PROPERTY_NAMES = {
    "shape",
    "input_shape",
    "output_shape",
    "object_shape",
    "unit_shape",
}

POSITION_PROPERTY_NAMES = {
    "position",
    "input_position",
    "output_position",
    "top_left",
    "center",
    "anchor_position",
    "top_left_positions",
    "positions",
    "cells",
}

SIGNED_NUMERIC_PROPERTY_NAMES = {
    "row_shift",
    "column_shift",
    "cell_count_delta",
    "height_delta",
    "width_delta",
    "row_delta",
    "column_delta",
    "distance_delta",
}

POSITIVE_MEASURE_PROPERTY_NAMES = {
    "distance",
    "row_spacing",
    "column_spacing",
    "spacing",
    "gap",
    "thickness",
    "height",
    "width",
    "input_height",
    "input_width",
    "output_height",
    "output_width",
    "scale_factor",
    "rotation_count",
}


def _freeze_value(value: Any) -> Any:
    """Convert mutable/nested values into deterministic hashable values."""
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                ((str(key), _freeze_value(item)) for key, item in value.items()),
                key=lambda pair: pair[0],
            )
        )

    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)

    if isinstance(value, set):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))

    try:
        hash(value)
    except TypeError:
        return repr(value)

    return value


def _properties_from_mapping(
    values: Mapping[str, Any] | None,
    **extra: Any,
) -> tuple[tuple[str, Any], ...]:
    merged: dict[str, Any] = {}

    if values:
        merged.update(values)

    merged.update(
        {
            name: value
            for name, value in extra.items()
            if value is not None
        }
    )

    return tuple(
        sorted(
            (
                (str(name), _freeze_value(value))
                for name, value in merged.items()
                if value is not None
            ),
            key=lambda item: item[0],
        )
    )


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


def _generalize_sequence(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_generalize_sequence(item) for item in value)
    if isinstance(value, list):
        return tuple(_generalize_sequence(item) for item in value)
    return value


def _generalize_property_value(name: str, value: Any) -> Any:
    """
    Generalize only for cross-pair matching.

    The original value always remains stored on the event.
    """
    lowered = name.lower()

    if lowered in COLOR_PROPERTY_NAMES or lowered.endswith("_color"):
        return "color"

    if lowered in DIMENSION_PROPERTY_NAMES or lowered.endswith("_dimensions"):
        return "dimensions"

    if lowered in SHAPE_PROPERTY_NAMES or lowered.endswith("_shape"):
        return "shape"

    if lowered in POSITION_PROPERTY_NAMES or lowered.endswith("_positions"):
        return "positions"

    if lowered in SIGNED_NUMERIC_PROPERTY_NAMES:
        if isinstance(value, (int, float)):
            return _generalize_number(value)

    if lowered in POSITIVE_MEASURE_PROPERTY_NAMES:
        if isinstance(value, (int, float)):
            return "measure"

    if lowered in COUNT_PROPERTY_NAMES or lowered.endswith("_count"):
        if isinstance(value, (int, float)):
            return "count"

    if lowered in {"translation_vector", "offset", "delta", "vector"}:
        if isinstance(value, tuple):
            return tuple(
                _generalize_number(item)
                if isinstance(item, (int, float))
                else item
                for item in value
            )

    # Preserve structural booleans and symbolic categories exactly.
    if isinstance(value, (bool, str)) or value is None:
        return value

    # Generic fallback prevents exact coordinates/numbers from blocking
    # cross-pair matches while retaining nested structural form.
    if isinstance(value, (int, float)):
        return "number"

    if isinstance(value, tuple):
        return tuple(
            "number" if isinstance(item, (int, float))
            else _generalize_sequence(item)
            for item in value
        )

    return value


# =============================================================================
# EVENT SPECIFICATIONS
# =============================================================================

@dataclass(frozen=True)
class EventSpec:
    event_type: str
    subject_type: str
    action: str
    object_type: str | None = None
    implied_properties: tuple[tuple[str, Any], ...] = ()


def _spec(
    event_type: str,
    subject_type: str,
    action: str,
    object_type: str | None = None,
    **implied_properties: Any,
) -> EventSpec:
    return EventSpec(
        event_type=event_type,
        subject_type=subject_type,
        action=action,
        object_type=object_type,
        implied_properties=_properties_from_mapping(implied_properties),
    )


# This registry is intentionally broad. New Pair Story categories can be added
# here without creating another generalizer function.
CATEGORY_SPECS: dict[str, EventSpec] = {
    # Grid / canvas
    "grid_shape_change": _spec("grid_shape_change", "grid", "resize_canvas"),
    "background_creation": _spec(
        "background_canvas_creation",
        "output_canvas",
        "create_background",
        new_identity=True,
    ),
    "background_removal": _spec(
        "background_canvas_deletion",
        "input_canvas",
        "delete_background",
        identity_removed=True,
    ),
    "background_change": _spec(
        "background_canvas_change",
        "canvas",
        "change_background",
    ),

    # Individual objects
    "added_object": _spec(
        "object_creation",
        "output_object",
        "create",
        new_identity=True,
    ),
    "removed_object": _spec(
        "object_deletion",
        "input_object",
        "delete",
        identity_removed=True,
    ),
    "movement": _spec(
        "object_translation",
        "matched_object",
        "move",
        preserves_identity=True,
    ),
    "color_change": _spec(
        "object_color_change",
        "matched_object",
        "recolor",
        preserves_identity=True,
    ),
    "size_change": _spec(
        "object_size_change",
        "matched_object",
        "resize",
        preserves_identity=True,
    ),
    "dimension_change": _spec(
        "object_dimension_change",
        "matched_object",
        "change_dimensions",
        preserves_identity=True,
    ),
    "shape_change": _spec(
        "object_shape_change",
        "matched_object",
        "reshape",
        preserves_identity=True,
    ),
    "role_change": _spec(
        "scene_role_change",
        "matched_object",
        "change_role",
        preserves_identity=True,
    ),
    "pattern_change": _spec(
        "internal_pattern_change",
        "matched_object",
        "change_cell_pattern",
        preserves_identity=True,
    ),
    "unchanged": _spec(
        "object_preservation",
        "matched_object",
        "preserve",
        preserves_identity=True,
    ),

    # Object collections
    "added_object_group": _spec(
        "object_collection_creation",
        "output_object_collection",
        "create_collection",
        new_identity=True,
    ),
    "removed_object_group": _spec(
        "object_collection_deletion",
        "input_object_collection",
        "delete_collection",
        identity_removed=True,
    ),
    "object_array_creation": _spec(
        "regular_array_creation",
        "output_object_array",
        "create_regular_array",
        new_identity=True,
        regular_layout=True,
    ),
    "object_array_removal": _spec(
        "regular_array_deletion",
        "input_object_array",
        "delete_regular_array",
        identity_removed=True,
        regular_layout=True,
    ),
    "object_repetition_creation": _spec(
        "regular_repetition_creation",
        "output_object_repetition",
        "create_regular_repetition",
        new_identity=True,
        regular_layout=True,
    ),
    "object_repetition_removal": _spec(
        "regular_repetition_deletion",
        "input_object_repetition",
        "delete_regular_repetition",
        identity_removed=True,
        regular_layout=True,
    ),

    # Relationships
    "relationship_reversal": _spec(
        "relationship_reversal",
        "matched_object",
        "reverse_relationship",
        "matched_object",
        directional=True,
    ),
    "relationship_added": _spec(
        "relationship_creation",
        "matched_object",
        "add_relationship",
        "matched_object",
    ),
    "relationship_removed": _spec(
        "relationship_deletion",
        "matched_object",
        "remove_relationship",
        "matched_object",
    ),

    # Explicit no-op
    "no_change": _spec("scene_preservation", "scene", "preserve"),
}


# =============================================================================
# SPECIAL PROPERTY ENRICHMENT
# =============================================================================

PropertyEnricher = Callable[[StoryStatement, dict[str, Any]], None]


def _enrich_movement(statement: StoryStatement, props: dict[str, Any]) -> None:
    row_shift = props.get("row_shift", 0)
    column_shift = props.get("column_shift", 0)

    if not isinstance(row_shift, (int, float)):
        row_shift = 0
    if not isinstance(column_shift, (int, float)):
        column_shift = 0

    props.setdefault(
        "row_direction",
        "up" if row_shift < 0 else "down" if row_shift > 0 else "none",
    )
    props.setdefault(
        "column_direction",
        "left" if column_shift < 0 else "right" if column_shift > 0 else "none",
    )
    props.setdefault("translation_vector", (row_shift, column_shift))
    props.setdefault("distance", abs(row_shift) + abs(column_shift))
    props.setdefault("movement_kind", "translation")


def _enrich_size_change(
    statement: StoryStatement,
    props: dict[str, Any],
) -> None:
    if "size_direction" in props:
        return

    text = statement.text.lower()
    if " grew " in f" {text} ":
        props["size_direction"] = "increase"
    elif " shrank " in f" {text} ":
        props["size_direction"] = "decrease"
    else:
        props["size_direction"] = "changed"


def _enrich_preservation(
    statement: StoryStatement,
    props: dict[str, Any],
) -> None:
    props.setdefault("position_preserved", True)
    props.setdefault("color_preserved", True)
    props.setdefault("size_preserved", True)
    props.setdefault("shape_preserved", True)


def _enrich_regular_array(
    statement: StoryStatement,
    props: dict[str, Any],
) -> None:
    props.setdefault("layout_kind", "regular_array")
    props.setdefault("regular_layout", True)
    props.setdefault("complete_lattice", True)

    dimensions = props.get("dimensions")
    if dimensions is not None:
        props.setdefault("object_dimensions", dimensions)


def _enrich_regular_repetition(
    statement: StoryStatement,
    props: dict[str, Any],
) -> None:
    props.setdefault("layout_kind", "regular_repetition")
    props.setdefault("regular_layout", True)

    dimensions = props.get("dimensions")
    if dimensions is not None:
        props.setdefault("object_dimensions", dimensions)


CATEGORY_ENRICHERS: dict[str, PropertyEnricher] = {
    "movement": _enrich_movement,
    "size_change": _enrich_size_change,
    "unchanged": _enrich_preservation,
    "object_array_creation": _enrich_regular_array,
    "object_array_removal": _enrich_regular_array,
    "object_repetition_creation": _enrich_regular_repetition,
    "object_repetition_removal": _enrich_regular_repetition,
}


# =============================================================================
# GENERIC FALLBACK FOR FUTURE CATEGORIES
# =============================================================================

def _infer_fallback_spec(category: str) -> EventSpec:
    """
    Preserve unknown future categories rather than dropping them.

    Naming conventions allow useful automatic behavior:
      *_creation / added_*   -> creation
      *_deletion / removed_* -> deletion
      *_change               -> change
      *_preservation         -> preserve
    """
    normalized = category.strip().lower() or "unknown_statement"

    if (
        normalized.endswith("_creation")
        or normalized.startswith("added_")
        or normalized.startswith("created_")
    ):
        return _spec(
            normalized,
            "output_structure",
            "create",
            new_identity=True,
        )

    if (
        normalized.endswith("_deletion")
        or normalized.endswith("_removal")
        or normalized.startswith("removed_")
        or normalized.startswith("deleted_")
    ):
        return _spec(
            normalized,
            "input_structure",
            "delete",
            identity_removed=True,
        )

    if normalized.endswith("_preservation") or normalized == "unchanged":
        return _spec(normalized, "matched_structure", "preserve")

    if normalized.endswith("_change"):
        return _spec(normalized, "matched_structure", "change")

    return _spec(
        f"unclassified_{normalized}",
        "scene_structure",
        "record",
        original_category=normalized,
    )


# =============================================================================
# STATEMENT GENERALIZATION
# =============================================================================

def generalize_statement(
    statement: StoryStatement,
) -> GeneralizedEvent:
    spec = CATEGORY_SPECS.get(
        statement.category,
        _infer_fallback_spec(statement.category),
    )

    props = dict(statement.properties)

    for name, value in spec.implied_properties:
        props.setdefault(name, value)

    enricher = CATEGORY_ENRICHERS.get(statement.category)
    if enricher is not None:
        enricher(statement, props)

    return GeneralizedEvent(
        event_type=spec.event_type,
        subject_type=spec.subject_type,
        action=spec.action,
        object_type=spec.object_type,
        properties=_properties_from_mapping(props),
        source_category=statement.category,
        source_statement=statement.text,
        input_object_ids=tuple(statement.input_object_ids),
        output_object_ids=tuple(statement.output_object_ids),
    )


# =============================================================================
# PAIR STORY GENERALIZATION
# =============================================================================

def _deduplicate_events(
    events: Iterable[GeneralizedEvent],
) -> tuple[GeneralizedEvent, ...]:
    """
    Remove only exact generalized duplicates.

    Events with different concrete properties, object IDs, counts, layouts,
    colors, dimensions, or source categories remain separate.
    """
    unique: list[GeneralizedEvent] = []
    seen: set[tuple[Any, ...]] = set()

    for event in events:
        key = (
            event.signature(include_values=True, include_source_category=True),
            event.input_object_ids,
            event.output_object_ids,
        )

        if key in seen:
            continue

        seen.add(key)
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
            statement.category in {"unchanged", "no_change"}
            and not include_preservations
        ):
            continue

        events.append(generalize_statement(statement))

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
    minimum_match_score: float = 0.60,
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

def _minimum_counter(
    counters: Sequence[Counter[Any]],
) -> Counter[Any]:
    if not counters:
        return Counter()

    common = counters[0].copy()

    for counter in counters[1:]:
        for key in list(common):
            common[key] = min(common[key], counter.get(key, 0))
            if common[key] <= 0:
                del common[key]

    return common


def build_generalized_common_story(
    stories: Sequence[GeneralizedPairStory],
) -> GeneralizedCommonStory:
    if not stories:
        return GeneralizedCommonStory(
            pair_count=0,
            shared_event_signatures=(),
            shared_event_types=(),
            pair_event_counts=(),
            shared_event_type_counts=(),
            shared_signature_counts=(),
        )

    type_counters = [
        Counter(story.event_types)
        for story in stories
    ]
    signature_counters = [
        Counter(story.signatures(include_values=False))
        for story in stories
    ]

    common_types = _minimum_counter(type_counters)
    common_signatures = _minimum_counter(signature_counters)

    return GeneralizedCommonStory(
        pair_count=len(stories),
        shared_event_signatures=tuple(
            sorted(common_signatures.keys(), key=repr)
        ),
        shared_event_types=tuple(
            sorted(common_types.keys())
        ),
        pair_event_counts=tuple(
            len(story.events)
            for story in stories
        ),
        shared_event_type_counts=tuple(
            sorted(common_types.items(), key=lambda item: item[0])
        ),
        shared_signature_counts=tuple(
            sorted(common_signatures.items(), key=lambda item: repr(item[0]))
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
        target = event.object_type or "scene"

        lines.append(f"{index}. {event.event_type}")
        lines.append(
            f"   {event.subject_type} --{event.action}--> {target}"
        )

        if event.source_category:
            lines.append(f"   source category: {event.source_category}")

        property_text = _format_properties(event.properties)
        if property_text:
            lines.append(f"   properties: {property_text}")

        if event.input_object_ids:
            lines.append(
                f"   input object ids: {event.input_object_ids}"
            )

        if event.output_object_ids:
            lines.append(
                f"   output object ids: {event.output_object_ids}"
            )

        if event.source_statement:
            lines.append(f"   source: {event.source_statement}")

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

    type_counts = dict(common.shared_event_type_counts)
    for event_type in common.shared_event_types:
        count = type_counts.get(event_type, 1)
        suffix = f" x{count}" if count > 1 else ""
        lines.append(f"  ✓ {event_type}{suffix}")

    lines.append("")

    if common.shared_event_signatures:
        lines.append("Fully shared generalized event signatures:")

        signature_counts = dict(common.shared_signature_counts)
        for signature in common.shared_event_signatures:
            count = signature_counts.get(signature, 1)
            suffix = f" x{count}" if count > 1 else ""
            lines.append(f"  ✓ {signature}{suffix}")
    else:
        lines.append(
            "No complete event signature is shared by every pair."
        )

    lines.extend([
        "",
        "Concrete values remain attached to each pair event.",
        "Shared signatures use generalized values only.",
        "This is factual generalization, not an executable learned rule.",
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