from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from vision.story_generalizer import (
    GeneralizedEvent,
    GeneralizedPairStory,
    build_generalized_common_story,
    generalize_grid_pair,
)


@dataclass(frozen=True)
class RuleEvidence:
    name: str
    value: Any
    support_count: int
    pair_count: int

    @property
    def support_ratio(self) -> float:
        return 0.0 if self.pair_count == 0 else self.support_count / self.pair_count


@dataclass(frozen=True)
class RuleSlot:
    slot_name: str
    event_type: str
    structural_signature: tuple[tuple[str, Any], ...]
    exact_properties: tuple[tuple[str, Any], ...]
    variable_properties: tuple[str, ...]
    support_count: int
    pair_count: int
    minimum_count_per_pair: int
    maximum_count_per_pair: int

    @property
    def support_ratio(self) -> float:
        return 0.0 if self.pair_count == 0 else self.support_count / self.pair_count


@dataclass(frozen=True)
class CandidateRule:
    name: str
    required_event_types: tuple[str, ...]
    required_event_signatures: tuple[tuple[Any, ...], ...]
    preserved_facts: tuple[str, ...]
    variable_facts: tuple[str, ...]
    evidence: tuple[RuleEvidence, ...]
    support_count: int
    pair_count: int
    consistency: float
    specificity: float
    simplicity: float
    confidence: float
    explanation: str
    required_event_slots: tuple[RuleSlot, ...] = ()

    @property
    def support_ratio(self) -> float:
        return 0.0 if self.pair_count == 0 else self.support_count / self.pair_count


@dataclass(frozen=True)
class LearnedRule:
    best_rule: CandidateRule | None
    candidates: tuple[CandidateRule, ...]
    pair_count: int
    warning: str | None = None


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted(((str(k), _freeze(v)) for k, v in value.items()), key=lambda x: x[0]))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze(v) for v in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _event_properties(event: GeneralizedEvent) -> dict[str, Any]:
    try:
        return dict(event.properties)
    except (TypeError, ValueError):
        return {}


def _events_by_type(story: GeneralizedPairStory) -> dict[str, list[GeneralizedEvent]]:
    grouped: dict[str, list[GeneralizedEvent]] = {}
    for event in story.events:
        grouped.setdefault(event.event_type, []).append(event)
    return grouped


def _all_property_names(events: Iterable[GeneralizedEvent]) -> set[str]:
    names: set[str] = set()
    for event in events:
        names.update(_event_properties(event))
    return names


STRUCTURAL_PROPERTY_PRIORITY = (
    "scene_role",
    "layout_kind",
    "shape",
    "object_shape",
    "dimensions",
    "object_dimensions",
    "cell_count",
    "complete_lattice",
    "regular_layout",
    "row_direction",
    "column_direction",
    "size_direction",
    "old_relationship",
    "new_relationship",
    "relationship",
    "input_role",
    "output_role",
)

PAIR_SPECIFIC_PROPERTIES = {
    "top_left_positions",
    "positions",
    "cells",
    "origin",
    "placement",
    "row_shift",
    "column_shift",
    "translation_vector",
    "distance",
    "height_delta",
    "width_delta",
    "input_shape",
    "output_shape",
}

COLOR_PROPERTIES = {
    "color",
    "input_color",
    "output_color",
    "from_color",
    "to_color",
    "background_color",
    "foreground_color",
}


def _structural_signature(
    event: GeneralizedEvent,
    *,
    include_color: bool,
) -> tuple[tuple[str, Any], ...]:
    props = _event_properties(event)
    result: list[tuple[str, Any]] = []

    for name in STRUCTURAL_PROPERTY_PRIORITY:
        if name in props:
            result.append((name, _freeze(props[name])))

    if include_color:
        for name in sorted(COLOR_PROPERTIES):
            if name in props:
                result.append((name, _freeze(props[name])))

    result.extend([
        ("subject_type", event.subject_type),
        ("action", event.action),
        ("object_type", event.object_type),
    ])
    return tuple(result)


def _pair_slot_groups(
    story: GeneralizedPairStory,
    event_type: str,
    *,
    include_color: bool,
) -> dict[tuple[tuple[str, Any], ...], list[GeneralizedEvent]]:
    groups: dict[tuple[tuple[str, Any], ...], list[GeneralizedEvent]] = {}
    for event in story.events:
        if event.event_type != event_type:
            continue
        key = _structural_signature(event, include_color=include_color)
        groups.setdefault(key, []).append(event)
    return groups


def _shared_slot_keys(
    stories: Sequence[GeneralizedPairStory],
    event_type: str,
    *,
    include_color: bool,
) -> tuple[tuple[tuple[str, Any], ...], ...]:
    if not stories:
        return ()

    key_sets = [
        set(_pair_slot_groups(story, event_type, include_color=include_color))
        for story in stories
    ]
    shared = set.intersection(*key_sets) if key_sets else set()
    return tuple(sorted(shared, key=repr))


def _common_exact_properties(
    event_lists: Sequence[Sequence[GeneralizedEvent]],
) -> tuple[tuple[str, Any], ...]:
    if not event_lists or any(not events for events in event_lists):
        return ()

    pair_common_names: list[set[str]] = []
    for events in event_lists:
        names = set.intersection(*[set(_event_properties(event)) for event in events])
        pair_common_names.append(names)

    common_names = set.intersection(*pair_common_names)
    exact: list[tuple[str, Any]] = []

    for name in sorted(common_names):
        if name in PAIR_SPECIFIC_PROPERTIES:
            continue

        pair_values: list[tuple[Any, ...]] = []
        for events in event_lists:
            values = tuple(sorted({_freeze(_event_properties(e)[name]) for e in events}, key=repr))
            pair_values.append(values)

        if pair_values and all(v == pair_values[0] for v in pair_values[1:]) and len(pair_values[0]) == 1:
            exact.append((name, pair_values[0][0]))

    return tuple(exact)


def _variable_property_names(
    event_lists: Sequence[Sequence[GeneralizedEvent]],
) -> tuple[str, ...]:
    names: set[str] = set()
    for events in event_lists:
        for event in events:
            names.update(_event_properties(event))

    variable: list[str] = []
    for name in sorted(names):
        values: list[Any] = []
        present = True

        for events in event_lists:
            pair_values = [
                _freeze(_event_properties(event)[name])
                for event in events
                if name in _event_properties(event)
            ]
            if not pair_values:
                present = False
                break
            values.extend(pair_values)

        if present and len(set(values)) > 1:
            variable.append(name)

    return tuple(variable)


def _slot_name(
    event_type: str,
    signature: tuple[tuple[str, Any], ...],
    index: int,
) -> str:
    props = dict(signature)
    pieces: list[str] = []

    for name in ("scene_role", "color", "shape", "object_shape", "object_dimensions", "dimensions", "layout_kind"):
        value = props.get(name)
        if value is not None:
            pieces.append(str(value))

    suffix = "_".join(
        piece.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "x")
        for piece in pieces
    )
    return f"{event_type}:{suffix or f'slot_{index}'}"


def _role_key(
    event: GeneralizedEvent,
    story: GeneralizedPairStory,
) -> tuple[Any, ...]:
    """
    Coarse functional identity for matching structures across train pairs.

    Shape, color, dimensions, cell count, and exact positions are deliberately
    excluded because those may vary while the structure keeps the same role.
    """
    props = _event_properties(event)

    positions = props.get("top_left_positions") or props.get("positions") or ()
    normalized_center = ("unknown", "unknown")
    normalized_extent = ("unknown", "unknown")
    border_role = "unknown"

    valid_positions = [
        (item[0], item[1])
        for item in positions
        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[0], int)
            and isinstance(item[1], int)
        )
    ]

    if valid_positions:
        output_height = max(1, story.output_shape[0])
        output_width = max(1, story.output_shape[1])

        rows = [row for row, _ in valid_positions]
        columns = [column for _, column in valid_positions]

        top = min(rows)
        left = min(columns)
        bottom = max(rows)
        right = max(columns)

        normalized_center = (
            round(((top + bottom) / 2) / max(1, output_height - 1), 1),
            round(((left + right) / 2) / max(1, output_width - 1), 1),
        )

        normalized_extent = (
            round((bottom - top + 1) / output_height, 1),
            round((right - left + 1) / output_width, 1),
        )

        touches = (
            top == 0,
            left == 0,
            bottom >= output_height - 1,
            right >= output_width - 1,
        )

        if sum(touches) >= 3:
            border_role = "outer"
        elif 0.3 <= normalized_center[0] <= 0.7 and 0.3 <= normalized_center[1] <= 0.7:
            border_role = "central"
        else:
            border_role = "intermediate"

    return (
        ("scene_role", props.get("scene_role", "unknown")),
        ("layout_kind", props.get("layout_kind", "unknown")),
        ("complete_lattice", props.get("complete_lattice", False)),
        ("regular_layout", props.get("regular_layout", False)),
        ("border_role", border_role),
        ("normalized_center", normalized_center),
        ("normalized_extent", normalized_extent),
        ("subject_type", event.subject_type),
        ("action", event.action),
        ("object_type", event.object_type),
    )


def _role_similarity(
    first_key: tuple[Any, ...],
    second_key: tuple[Any, ...],
) -> float:
    first = dict(first_key)
    second = dict(second_key)

    weights = {
        "scene_role": 3.0,
        "layout_kind": 2.5,
        "complete_lattice": 1.5,
        "regular_layout": 1.0,
        "border_role": 2.5,
        "normalized_center": 1.5,
        "normalized_extent": 1.5,
        "subject_type": 1.0,
        "action": 1.0,
        "object_type": 0.5,
    }

    total = 0.0
    maximum = 0.0

    for name, weight in weights.items():
        maximum += weight
        first_value = first.get(name)
        second_value = second.get(name)

        if first_value == second_value:
            total += weight
        elif (
            name in {"normalized_center", "normalized_extent"}
            and isinstance(first_value, tuple)
            and isinstance(second_value, tuple)
            and len(first_value) == 2
            and len(second_value) == 2
            and all(isinstance(value, (int, float)) for value in first_value + second_value)
        ):
            distance = (
                abs(first_value[0] - second_value[0])
                + abs(first_value[1] - second_value[1])
            )

            if distance <= 0.2:
                total += weight * 0.8
            elif distance <= 0.4:
                total += weight * 0.4

    return 0.0 if maximum == 0 else total / maximum


def _match_role_events(
    stories: Sequence[GeneralizedPairStory],
    event_type: str,
) -> list[list[GeneralizedEvent]]:
    """
    Match events one-to-one across pairs by functional role.

    The train pair containing the fewest events seeds the roles. Each other
    pair contributes the highest-scoring unmatched event to each role.
    """
    if not stories:
        return []

    pair_events = [
        [
            event
            for event in story.events
            if event.event_type == event_type
        ]
        for story in stories
    ]

    if any(not events for events in pair_events):
        return []

    seed_index = min(
        range(len(stories)),
        key=lambda index: len(pair_events[index]),
    )

    roles = [
        [event]
        for event in pair_events[seed_index]
    ]

    seed_keys = [
        _role_key(event, stories[seed_index])
        for event in pair_events[seed_index]
    ]

    for pair_index, events in enumerate(pair_events):
        if pair_index == seed_index:
            continue

        available = set(range(len(events)))
        assignments: dict[int, int] = {}

        candidates: list[tuple[float, int, int]] = []

        for role_index, seed_key in enumerate(seed_keys):
            for event_index, event in enumerate(events):
                candidates.append(
                    (
                        _role_similarity(
                            seed_key,
                            _role_key(event, stories[pair_index]),
                        ),
                        role_index,
                        event_index,
                    )
                )

        candidates.sort(reverse=True)

        used_roles: set[int] = set()

        for score, role_index, event_index in candidates:
            if score < 0.60:
                continue
            if role_index in used_roles:
                continue
            if event_index not in available:
                continue

            assignments[role_index] = event_index
            used_roles.add(role_index)
            available.remove(event_index)

        for role_index, role in enumerate(roles):
            event_index = assignments.get(role_index)

            if event_index is not None:
                role.append(events[event_index])

    return [
        role
        for role in roles
        if len(role) == len(stories)
    ]


def _role_slot_name(
    event_type: str,
    role_events: Sequence[GeneralizedEvent],
    stories: Sequence[GeneralizedPairStory],
    index: int,
) -> str:
    keys = [
        dict(_role_key(event, story))
        for event, story in zip(role_events, stories)
    ]

    common: dict[str, Any] = {}

    for name in keys[0]:
        values = [key.get(name) for key in keys]

        if all(value == values[0] for value in values[1:]):
            common[name] = values[0]

    pieces = [
        str(common[name])
        for name in ("scene_role", "layout_kind", "border_role")
        if common.get(name) not in {None, "unknown"}
    ]

    suffix = "_".join(pieces)

    return (
        f"{event_type}:{suffix}"
        if suffix
        else f"{event_type}:role_{index}"
    )


def _learn_slots_for_event_type(
    stories: Sequence[GeneralizedPairStory],
    event_type: str,
) -> tuple[RuleSlot, ...]:
    matched_roles = _match_role_events(
        stories,
        event_type,
    )

    slots: list[RuleSlot] = []

    for index, role_events in enumerate(
        matched_roles,
        start=1,
    ):
        event_lists = [
            [event]
            for event in role_events
        ]

        structural_signature = tuple(
            sorted(
                {
                    item
                    for event, story in zip(role_events, stories)
                    for item in _role_key(event, story)
                },
                key=lambda item: item[0],
            )
        )

        slots.append(
            RuleSlot(
                slot_name=_role_slot_name(
                    event_type,
                    role_events,
                    stories,
                    index,
                ),
                event_type=event_type,
                structural_signature=structural_signature,
                exact_properties=_common_exact_properties(
                    event_lists
                ),
                variable_properties=_variable_property_names(
                    event_lists
                ),
                support_count=len(stories),
                pair_count=len(stories),
                minimum_count_per_pair=1,
                maximum_count_per_pair=1,
            )
        )

    return tuple(slots)


def _learn_required_slots(
    stories: Sequence[GeneralizedPairStory],
    required_event_types: Sequence[str],
) -> tuple[RuleSlot, ...]:
    slots: list[RuleSlot] = []
    for event_type in required_event_types:
        slots.extend(_learn_slots_for_event_type(stories, event_type))
    return tuple(slots)


PRESERVATION_PROPERTY_NAMES = {
    "preserves_identity",
    "position_preserved",
    "color_preserved",
    "size_preserved",
    "shape_preserved",
}

VARIABLE_PROPERTY_NAMES = {
    "row_shift",
    "column_shift",
    "cell_count_delta",
    "height_delta",
    "width_delta",
    "row_direction",
    "column_direction",
    "size_direction",
    "row_count",
    "column_count",
    "row_spacing",
    "column_spacing",
    "count",
    "cell_count",
    "dimensions",
    "object_dimensions",
    "shape",
    "color",
    "scene_role",
}


def _learn_preserved_facts(
    stories: Sequence[GeneralizedPairStory],
    required_event_types: Sequence[str],
) -> tuple[str, ...]:
    facts: list[str] = []

    for event_type in dict.fromkeys(required_event_types):
        event_lists = [
            _events_by_type(story).get(event_type, [])
            for story in stories
        ]

        if any(not event_list for event_list in event_lists):
            continue

        property_names = set.intersection(
            *[_all_property_names(event_list) for event_list in event_lists]
        )

        for property_name in sorted(property_names):
            if property_name not in PRESERVATION_PROPERTY_NAMES:
                continue

            values_per_pair: list[set[Any]] = []

            for event_list in event_lists:
                values_per_pair.append({
                    _freeze(_event_properties(event).get(property_name))
                    for event in event_list
                    if property_name in _event_properties(event)
                })

            if all(values == {True} for values in values_per_pair):
                facts.append(f"{event_type}.{property_name}=True")

    return tuple(dict.fromkeys(facts))


def _learn_variable_facts(
    stories: Sequence[GeneralizedPairStory],
    required_event_types: Sequence[str],
    slots: Sequence[RuleSlot],
) -> tuple[str, ...]:
    facts: list[str] = []

    for slot in slots:
        for name in slot.variable_properties:
            if name in VARIABLE_PROPERTY_NAMES:
                facts.append(f"{slot.slot_name}.{name} varies")

    if "object_translation" in required_event_types:
        translation_distances: list[Any] = []

        for story in stories:
            for event in story.events:
                if event.event_type != "object_translation":
                    continue

                props = _event_properties(event)

                if "distance" in props:
                    translation_distances.append(
                        _freeze(props["distance"])
                    )

        if len(set(translation_distances)) > 1:
            facts.append(
                "moving_object.translation_distance varies"
            )

    return tuple(dict.fromkeys(facts))


def _learn_evidence(
    stories: Sequence[GeneralizedPairStory],
    required_event_types: Sequence[str],
    slots: Sequence[RuleSlot],
) -> tuple[RuleEvidence, ...]:
    pair_count = len(stories)
    evidence: list[RuleEvidence] = []

    for event_type in dict.fromkeys(required_event_types):
        support_count = sum(
            1
            for story in stories
            if event_type in story.event_types
        )
        evidence.append(
            RuleEvidence(
                name="event_type",
                value=event_type,
                support_count=support_count,
                pair_count=pair_count,
            )
        )

    for slot in slots:
        evidence.append(
            RuleEvidence(
                name="event_slot",
                value=slot.slot_name,
                support_count=slot.support_count,
                pair_count=slot.pair_count,
            )
        )

    return tuple(evidence)


EVENT_TYPE_NAMES = {
    "object_translation": "Translate matched object",
    "object_preservation": "Preserve matched object",
    "object_color_change": "Recolor matched object",
    "object_recoloring": "Recolor matched object",
    "object_size_change": "Resize matched object",
    "object_resize": "Resize matched object",
    "object_dimension_change": "Change object dimensions",
    "object_shape_change": "Reshape matched object",
    "scene_role_change": "Change scene role",
    "internal_pattern_change": "Change internal pattern",
    "object_creation": "Create object",
    "object_deletion": "Delete object",
    "object_collection_creation": "Create object collection",
    "object_collection_deletion": "Delete object collection",
    "regular_array_creation": "Create regular array",
    "regular_array_deletion": "Delete regular array",
    "regular_repetition_creation": "Create regular repetition",
    "regular_repetition_deletion": "Delete regular repetition",
    "background_canvas_creation": "Create background canvas",
    "background_canvas_deletion": "Delete background canvas",
    "background_canvas_change": "Change background canvas",
    "relationship_reversal": "Reverse spatial relationship",
    "relationship_creation": "Create relationship",
    "relationship_deletion": "Delete relationship",
    "relationship_change": "Change relationship",
    "grid_shape_change": "Resize output grid",
    "grid_resize": "Resize output grid",
    "pattern_expansion": "Expand pattern",
    "pattern_completion": "Complete pattern",
    "pattern_repair": "Repair pattern",
    "motif_creation": "Create motif",
    "motif_projection": "Project motif",
    "region_creation": "Create region",
    "region_deletion": "Delete region",
}


def _rule_name(
    required_event_types: Sequence[str],
    slots: Sequence[RuleSlot],
) -> str:
    meaningful = [
        event_type
        for event_type in dict.fromkeys(required_event_types)
        if event_type != "object_preservation"
    ]

    if not meaningful:
        return "Preserve the scene"

    readable: list[str] = []

    for event_type in meaningful:
        slot_count = sum(
            1
            for slot in slots
            if slot.event_type == event_type
        )

        name = EVENT_TYPE_NAMES.get(
            event_type,
            event_type.replace("_", " ").title(),
        )

        if slot_count > 1:
            name += f" x{slot_count}"

        readable.append(name)

    return " + ".join(readable)


def _candidate_support_count(
    stories: Sequence[GeneralizedPairStory],
    required_event_types: Sequence[str],
    slots: Sequence[RuleSlot],
) -> int:
    required = set(required_event_types)

    type_support = sum(
        1
        for story in stories
        if required.issubset(set(story.event_types))
    )

    if not slots:
        return type_support

    return min(
        type_support,
        min(slot.support_count for slot in slots),
    )


def _score_consistency(support_count: int, pair_count: int) -> float:
    return 0.0 if pair_count == 0 else support_count / pair_count


def _score_specificity(
    required_event_types: Sequence[str],
    required_event_signatures: Sequence[tuple[Any, ...]],
    slots: Sequence[RuleSlot],
) -> float:
    if not required_event_types:
        return 0.0

    return min(
        1.0,
        min(0.45, len(set(required_event_types)) * 0.11)
        + min(0.20, len(required_event_signatures) * 0.04)
        + min(0.35, len(slots) * 0.055),
    )


def _score_simplicity(
    required_event_types: Sequence[str],
    variable_facts: Sequence[str],
    slots: Sequence[RuleSlot],
) -> float:
    complexity = (
        len(set(required_event_types))
        + 0.45 * len(slots)
        + 0.25 * len(variable_facts)
    )
    return 1.0 / (1.0 + 0.14 * max(0.0, complexity - 1.0))


def _score_confidence(
    consistency: float,
    specificity: float,
    simplicity: float,
    pair_count: int,
) -> float:
    evidence_strength = min(1.0, pair_count / 3.0)
    return max(
        0.0,
        min(
            1.0,
            0.52 * consistency
            + 0.23 * specificity
            + 0.15 * simplicity
            + 0.10 * evidence_strength,
        ),
    )


def _humanize_fact(fact: str) -> str:
    return (
        fact.replace(".", ": ")
        .replace("_", " ")
        .replace("=True", " is preserved")
    )


def _format_properties(
    properties: Sequence[tuple[str, Any]],
) -> str:
    if not properties:
        return "(none fixed)"
    return ", ".join(f"{name}={value!r}" for name, value in properties)


def _build_explanation(
    name: str,
    required_event_types: Sequence[str],
    slots: Sequence[RuleSlot],
    preserved_facts: Sequence[str],
    variable_facts: Sequence[str],
    support_count: int,
    pair_count: int,
) -> str:
    lines = [
        f"Rule: {name}",
        "",
        "Required transformations:",
    ]

    for event_type in dict.fromkeys(required_event_types):
        lines.append(
            "  - "
            + EVENT_TYPE_NAMES.get(
                event_type,
                event_type.replace("_", " "),
            )
        )

    if slots:
        lines.extend(["", "Separately learned structural roles:"])

        for slot in slots:
            multiplicity = (
                str(slot.minimum_count_per_pair)
                if slot.minimum_count_per_pair == slot.maximum_count_per_pair
                else f"{slot.minimum_count_per_pair}..{slot.maximum_count_per_pair}"
            )

            lines.extend([
                f"  - {slot.slot_name}",
                f"      event: {slot.event_type}",
                f"      occurrences per pair: {multiplicity}",
                f"      fixed: {_format_properties(slot.exact_properties)}",
            ])

            if slot.variable_properties:
                lines.append(
                    "      variable: "
                    + ", ".join(slot.variable_properties)
                )

    if preserved_facts:
        lines.extend(["", "Facts preserved across all train pairs:"])
        lines.extend(f"  - {_humanize_fact(fact)}" for fact in preserved_facts)

    if variable_facts:
        lines.extend(["", "Values allowed to vary:"])
        lines.extend(f"  - {_humanize_fact(fact)}" for fact in variable_facts)

    lines.extend(["", f"Support: {support_count} / {pair_count} train pairs"])
    return "\n".join(lines)


def _build_candidate(
    stories: Sequence[GeneralizedPairStory],
    required_event_types: Sequence[str],
    required_event_signatures: Sequence[tuple[Any, ...]],
) -> CandidateRule:
    pair_count = len(stories)
    unique_types = tuple(dict.fromkeys(required_event_types))
    slots = _learn_required_slots(stories, unique_types)
    support_count = _candidate_support_count(stories, unique_types, slots)
    preserved_facts = _learn_preserved_facts(stories, unique_types)
    variable_facts = _learn_variable_facts(stories, unique_types, slots)
    evidence = _learn_evidence(stories, unique_types, slots)

    consistency = _score_consistency(support_count, pair_count)
    specificity = _score_specificity(
        unique_types,
        required_event_signatures,
        slots,
    )
    simplicity = _score_simplicity(
        unique_types,
        variable_facts,
        slots,
    )
    confidence = _score_confidence(
        consistency,
        specificity,
        simplicity,
        pair_count,
    )

    name = _rule_name(unique_types, slots)

    return CandidateRule(
        name=name,
        required_event_types=unique_types,
        required_event_signatures=tuple(required_event_signatures),
        preserved_facts=preserved_facts,
        variable_facts=variable_facts,
        evidence=evidence,
        support_count=support_count,
        pair_count=pair_count,
        consistency=consistency,
        specificity=specificity,
        simplicity=simplicity,
        confidence=confidence,
        explanation=_build_explanation(
            name,
            unique_types,
            slots,
            preserved_facts,
            variable_facts,
            support_count,
            pair_count,
        ),
        required_event_slots=slots,
    )


def _candidate_sort_key(
    candidate: CandidateRule,
) -> tuple[float, float, float, float, int]:
    return (
        candidate.confidence,
        candidate.consistency,
        candidate.specificity,
        candidate.simplicity,
        -len(candidate.required_event_types),
    )


def _event_type_counts(
    stories: Sequence[GeneralizedPairStory],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for story in stories:
        counts.update(set(story.event_types))
    return counts


def learn_rule(
    stories: Sequence[GeneralizedPairStory],
) -> LearnedRule:
    stories = tuple(stories)
    pair_count = len(stories)

    if pair_count == 0:
        return LearnedRule(
            best_rule=None,
            candidates=(),
            pair_count=0,
            warning="No generalized train stories were provided.",
        )

    common = build_generalized_common_story(stories)
    shared_types = tuple(common.shared_event_types)
    shared_signatures = tuple(common.shared_event_signatures)

    candidates: list[CandidateRule] = []

    if shared_types:
        candidates.append(
            _build_candidate(
                stories,
                shared_types,
                shared_signatures,
            )
        )

    meaningful_shared_types = tuple(
        event_type
        for event_type in shared_types
        if event_type != "object_preservation"
    )

    if meaningful_shared_types and meaningful_shared_types != shared_types:
        candidates.append(
            _build_candidate(
                stories,
                meaningful_shared_types,
                tuple(
                    signature
                    for signature in shared_signatures
                    if signature[0] != "object_preservation"
                ),
            )
        )

    type_counts = _event_type_counts(stories)

    near_common_types = tuple(
        sorted(
            event_type
            for event_type, count in type_counts.items()
            if (
                count >= max(1, pair_count - 1)
                and event_type not in shared_types
            )
        )
    )

    if near_common_types:
        candidates.append(
            _build_candidate(
                stories,
                near_common_types,
                (),
            )
        )

    if not candidates:
        return LearnedRule(
            best_rule=None,
            candidates=(),
            pair_count=pair_count,
            warning=(
                "No transformation event was shared strongly "
                "enough across the train pairs to form a rule."
            ),
        )

    unique: dict[
        tuple[tuple[str, ...], tuple[str, ...]],
        CandidateRule,
    ] = {}

    for candidate in candidates:
        key = (
            candidate.required_event_types,
            tuple(
                slot.slot_name
                for slot in candidate.required_event_slots
            ),
        )

        existing = unique.get(key)

        if existing is None or candidate.confidence > existing.confidence:
            unique[key] = candidate

    sorted_candidates = tuple(
        sorted(
            unique.values(),
            key=_candidate_sort_key,
            reverse=True,
        )
    )

    best_rule = sorted_candidates[0]
    warning = None

    if best_rule.support_ratio < 1.0:
        warning = (
            "The best structural rule is not supported "
            "by every train pair."
        )
    elif pair_count == 1:
        warning = (
            "Only one train pair is available, so the rule "
            "is descriptive but not strongly validated."
        )

    return LearnedRule(
        best_rule=best_rule,
        candidates=sorted_candidates,
        pair_count=pair_count,
        warning=warning,
    )


def format_candidate_rule(
    candidate: CandidateRule,
    *,
    index: int | None = None,
) -> str:
    title = (
        f"CANDIDATE {index}: {candidate.name}"
        if index is not None
        else candidate.name
    )

    return "\n".join([
        title,
        "-" * len(title),
        f"Confidence : {candidate.confidence:.3f}",
        f"Consistency: {candidate.consistency:.3f}",
        f"Specificity: {candidate.specificity:.3f}",
        f"Simplicity : {candidate.simplicity:.3f}",
        f"Support    : {candidate.support_count}/{candidate.pair_count}",
        f"Event slots: {len(candidate.required_event_slots)}",
        "",
        candidate.explanation,
    ])


def format_learned_rule(
    learned: LearnedRule,
    *,
    max_candidates: int = 3,
) -> str:
    lines = [
        "=" * 72,
        "LEARNED RULE",
        "=" * 72,
        f"Train pairs: {learned.pair_count}",
        "",
    ]

    if learned.best_rule is None:
        lines.append(
            learned.warning
            or "No rule could be learned."
        )
        return "\n".join(lines)

    lines.extend([
        "BEST RULE",
        "-" * 72,
        format_candidate_rule(learned.best_rule),
    ])

    if learned.warning:
        lines.extend([
            "",
            "WARNING",
            "-" * 72,
            learned.warning,
        ])

    remaining = learned.candidates[1:max_candidates]

    if remaining:
        lines.extend([
            "",
            "OTHER CANDIDATES",
            "-" * 72,
        ])

        for index, candidate in enumerate(
            remaining,
            start=2,
        ):
            lines.extend([
                "",
                format_candidate_rule(
                    candidate,
                    index=index,
                ),
            ])

    lines.extend([
        "",
        "This rule is descriptive only.",
        (
            "Structural roles are retained separately "
            "for planning and execution."
        ),
        (
            "It has not yet been executed against the "
            "train or test grids."
        ),
    ])

    return "\n".join(lines)


def learn_rule_from_grid_pairs(
    train_pairs: Sequence[
        tuple[
            Sequence[Sequence[int]],
            Sequence[Sequence[int]],
        ]
    ],
    *,
    connectivity: int = 4,
    include_preservations: bool = True,
) -> LearnedRule:
    stories = [
        generalize_grid_pair(
            input_grid,
            output_grid,
            connectivity=connectivity,
            include_preservations=include_preservations,
        )
        for input_grid, output_grid in train_pairs
    ]

    return learn_rule(stories)


def _shift_right(
    grid: list[list[int]],
    color: int,
    amount: int,
) -> list[list[int]]:
    output = [row[:] for row in grid]
    cells: list[tuple[int, int]] = []

    for row_index, row in enumerate(grid):
        for column_index, value in enumerate(row):
            if value == color:
                cells.append((row_index, column_index))
                output[row_index][column_index] = 0

    for row_index, column_index in cells:
        output[row_index][column_index + amount] = color

    return output


def _self_test() -> None:
    input_1 = [
        [0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 0],
        [0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    input_2 = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 6, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]

    output_1 = _shift_right(
        input_1,
        color=2,
        amount=3,
    )
    output_2 = _shift_right(
        input_2,
        color=4,
        amount=2,
    )

    learned = learn_rule_from_grid_pairs([
        (input_1, output_1),
        (input_2, output_2),
    ])

    print(format_learned_rule(learned))


if __name__ == "__main__":
    _self_test()