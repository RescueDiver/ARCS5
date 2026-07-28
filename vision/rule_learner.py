from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from vision.story_generalizer import (
    GeneralizedEvent,
    GeneralizedPairStory,
    build_generalized_common_story,
    generalize_grid_pair,
)


# =============================================================================
# LEARNED RULE DATA TYPES
# =============================================================================

@dataclass(frozen=True)
class RuleEvidence:
    name: str
    value: Any
    support_count: int
    pair_count: int

    @property
    def support_ratio(self) -> float:
        if self.pair_count == 0:
            return 0.0

        return self.support_count / self.pair_count


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

    @property
    def support_ratio(self) -> float:
        if self.pair_count == 0:
            return 0.0

        return self.support_count / self.pair_count


@dataclass(frozen=True)
class LearnedRule:
    best_rule: CandidateRule | None
    candidates: tuple[CandidateRule, ...]
    pair_count: int
    warning: str | None = None


# =============================================================================
# EVENT HELPERS
# =============================================================================

def _event_properties(event: GeneralizedEvent) -> dict[str, Any]:
    return dict(event.properties)


def _events_by_type(
    story: GeneralizedPairStory,
) -> dict[str, list[GeneralizedEvent]]:
    grouped: dict[str, list[GeneralizedEvent]] = {}

    for event in story.events:
        grouped.setdefault(event.event_type, []).append(event)

    return grouped


def _event_type_counts(
    stories: Sequence[GeneralizedPairStory],
) -> Counter[str]:
    counts: Counter[str] = Counter()

    for story in stories:
        counts.update(set(story.event_types))

    return counts


def _signature_counts(
    stories: Sequence[GeneralizedPairStory],
) -> Counter[tuple[Any, ...]]:
    counts: Counter[tuple[Any, ...]] = Counter()

    for story in stories:
        counts.update(set(story.signatures(include_values=False)))

    return counts


def _all_property_names(
    events: Iterable[GeneralizedEvent],
) -> set[str]:
    names: set[str] = set()

    for event in events:
        names.update(name for name, _ in event.properties)

    return names


# =============================================================================
# FACT LEARNING
# =============================================================================

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
}


def _learn_preserved_facts(
    stories: Sequence[GeneralizedPairStory],
    required_event_types: Sequence[str],
) -> tuple[str, ...]:
    """
    Learn preservation facts without mixing different object roles.

    A translated object and an unchanged object are treated as separate roles.
    This prevents facts such as "position preserved" from being assigned to
    the moving object.
    """
    if not stories:
        return ()

    facts: list[str] = []

    if "object_translation" in required_event_types:
        translation_events = [
            event
            for story in stories
            for event in _events_by_type(story).get(
                "object_translation",
                [],
            )
        ]

        if translation_events:
            if all(
                _event_properties(event).get(
                    "preserves_identity"
                ) is True
                for event in translation_events
            ):
                facts.append(
                    "moving_object.identity_preserved"
                )

            facts.extend([
                "moving_object.shape_preserved",
                "moving_object.color_preserved",
                "moving_object.size_preserved",
            ])

    if "object_preservation" in required_event_types:
        preservation_events = [
            event
            for story in stories
            for event in _events_by_type(story).get(
                "object_preservation",
                [],
            )
        ]

        if preservation_events:
            facts.append(
                "stationary_object.remains_unchanged"
            )

    for event_type in required_event_types:
        if event_type in {
            "object_translation",
            "object_preservation",
        }:
            continue

        event_lists = [
            _events_by_type(story).get(event_type, [])
            for story in stories
        ]

        if any(not event_list for event_list in event_lists):
            continue

        property_names = set.intersection(
            *[
                _all_property_names(event_list)
                for event_list in event_lists
            ]
        )

        for property_name in sorted(property_names):
            if property_name not in PRESERVATION_PROPERTY_NAMES:
                continue

            values_per_pair: list[set[Any]] = []

            for event_list in event_lists:
                pair_values = {
                    _event_properties(event).get(property_name)
                    for event in event_list
                    if property_name in _event_properties(event)
                }
                values_per_pair.append(pair_values)

            if all(values == {True} for values in values_per_pair):
                facts.append(
                    f"{event_type}.{property_name}"
                )

    return tuple(dict.fromkeys(facts))


def _learn_variable_facts(
    stories: Sequence[GeneralizedPairStory],
    required_event_types: Sequence[str],
) -> tuple[str, ...]:
    if not stories:
        return ()

    variable_facts: list[str] = []

    if "object_translation" in required_event_types:
        row_directions: list[Any] = []
        column_directions: list[Any] = []

        for story in stories:
            for event in _events_by_type(story).get(
                "object_translation",
                [],
            ):
                properties = _event_properties(event)

                if "row_direction" in properties:
                    row_directions.append(
                        properties["row_direction"]
                    )

                if "column_direction" in properties:
                    column_directions.append(
                        properties["column_direction"]
                    )

        if len(set(row_directions)) > 1:
            variable_facts.append(
                "moving_object.row_direction varies"
            )

        if len(set(column_directions)) > 1:
            variable_facts.append(
                "moving_object.column_direction varies"
            )

        # PairStory currently reports direction but not a structured exact
        # distance property, so distance remains an explicit learned variable.
        variable_facts.append(
            "moving_object.translation_distance varies"
        )

    for event_type in required_event_types:
        if event_type == "object_translation":
            continue

        collected: dict[str, list[Any]] = {}

        for story in stories:
            for event in _events_by_type(story).get(event_type, []):
                for name, value in event.properties:
                    if name in VARIABLE_PROPERTY_NAMES:
                        collected.setdefault(name, []).append(value)

        for name, values in sorted(collected.items()):
            if len(set(values)) > 1:
                variable_facts.append(
                    f"{event_type}.{name} varies"
                )

    return tuple(dict.fromkeys(variable_facts))


def _learn_evidence(
    stories: Sequence[GeneralizedPairStory],
    required_event_types: Sequence[str],
) -> tuple[RuleEvidence, ...]:
    pair_count = len(stories)
    evidence: list[RuleEvidence] = []

    for event_type in required_event_types:
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

    return tuple(evidence)


# =============================================================================
# RULE NAMING
# =============================================================================

EVENT_TYPE_NAMES = {
    "object_translation": "Translate matched object",
    "object_preservation": "Preserve matched object",
    "object_color_change": "Recolor matched object",
    "object_size_change": "Resize matched object",
    "object_dimension_change": "Change object dimensions",
    "object_shape_change": "Reshape matched object",
    "scene_role_change": "Change scene role",
    "internal_pattern_change": "Change internal pattern",
    "object_creation": "Create object",
    "object_deletion": "Delete object",
    "relationship_reversal": "Reverse spatial relationship",
    "relationship_creation": "Create relationship",
    "relationship_deletion": "Delete relationship",
    "grid_shape_change": "Resize output grid",
}


def _rule_name(
    required_event_types: Sequence[str],
) -> str:
    meaningful = [
        event_type
        for event_type in required_event_types
        if event_type != "object_preservation"
    ]

    if not meaningful:
        return "Preserve the scene"

    if (
        meaningful == ["object_translation"]
        and "object_preservation" in required_event_types
    ):
        return "Translate one object while preserving the rest"

    if len(meaningful) == 1:
        return EVENT_TYPE_NAMES.get(
            meaningful[0],
            meaningful[0].replace("_", " ").title(),
        )

    readable = [
        EVENT_TYPE_NAMES.get(
            event_type,
            event_type.replace("_", " ").title(),
        )
        for event_type in meaningful
    ]

    return " + ".join(readable)


# =============================================================================
# RULE SCORING
# =============================================================================

def _score_consistency(
    support_count: int,
    pair_count: int,
) -> float:
    if pair_count == 0:
        return 0.0

    return support_count / pair_count


def _score_specificity(
    required_event_types: Sequence[str],
    required_event_signatures: Sequence[tuple[Any, ...]],
) -> float:
    if not required_event_types:
        return 0.0

    signature_bonus = min(
        0.35,
        len(required_event_signatures) * 0.07,
    )

    event_score = min(
        0.65,
        len(required_event_types) * 0.18,
    )

    return min(1.0, event_score + signature_bonus)


def _score_simplicity(
    required_event_types: Sequence[str],
    variable_facts: Sequence[str],
) -> float:
    complexity = (
        len(required_event_types)
        + 0.5 * len(variable_facts)
    )

    return 1.0 / (1.0 + 0.18 * max(0.0, complexity - 1.0))


def _score_confidence(
    consistency: float,
    specificity: float,
    simplicity: float,
    pair_count: int,
) -> float:
    evidence_strength = min(1.0, pair_count / 3.0)

    confidence = (
        0.55 * consistency
        + 0.20 * specificity
        + 0.15 * simplicity
        + 0.10 * evidence_strength
    )

    return max(0.0, min(1.0, confidence))


# =============================================================================
# EXPLANATION
# =============================================================================

def _humanize_fact(fact: str) -> str:
    replacements = {
        "moving_object.identity_preserved":
            "moving object preserves identity",
        "moving_object.shape_preserved":
            "moving object preserves shape",
        "moving_object.color_preserved":
            "moving object preserves color",
        "moving_object.size_preserved":
            "moving object preserves size",
        "stationary_object.remains_unchanged":
            "other matched objects remain unchanged",
        "moving_object.row_direction varies":
            "vertical direction may vary",
        "moving_object.column_direction varies":
            "horizontal direction may vary",
        "moving_object.translation_distance varies":
            "translation distance may vary",
    }

    if fact in replacements:
        return replacements[fact]

    return (
        fact.replace(".", ": ")
        .replace("_", " ")
    )


def _build_explanation(
    name: str,
    required_event_types: Sequence[str],
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

    for event_type in required_event_types:
        lines.append(
            "  - "
            + EVENT_TYPE_NAMES.get(
                event_type,
                event_type.replace("_", " "),
            )
        )

    if preserved_facts:
        lines.extend([
            "",
            "Facts preserved across all train pairs:",
        ])

        for fact in preserved_facts:
            lines.append(f"  - {_humanize_fact(fact)}")

    if variable_facts:
        lines.extend([
            "",
            "Values allowed to vary:",
        ])

        for fact in variable_facts:
            lines.append(f"  - {_humanize_fact(fact)}")

    lines.extend([
        "",
        f"Support: {support_count} / {pair_count} train pairs",
    ])

    return "\n".join(lines)


# =============================================================================
# CANDIDATE CONSTRUCTION
# =============================================================================

def _build_candidate(
    stories: Sequence[GeneralizedPairStory],
    required_event_types: Sequence[str],
    required_event_signatures: Sequence[tuple[Any, ...]],
) -> CandidateRule:
    pair_count = len(stories)

    support_count = sum(
        1
        for story in stories
        if all(
            event_type in story.event_types
            for event_type in required_event_types
        )
    )

    preserved_facts = _learn_preserved_facts(
        stories,
        required_event_types,
    )

    variable_facts = _learn_variable_facts(
        stories,
        required_event_types,
    )

    evidence = _learn_evidence(
        stories,
        required_event_types,
    )

    consistency = _score_consistency(
        support_count,
        pair_count,
    )

    specificity = _score_specificity(
        required_event_types,
        required_event_signatures,
    )

    simplicity = _score_simplicity(
        required_event_types,
        variable_facts,
    )

    confidence = _score_confidence(
        consistency,
        specificity,
        simplicity,
        pair_count,
    )

    name = _rule_name(required_event_types)

    explanation = _build_explanation(
        name,
        required_event_types,
        preserved_facts,
        variable_facts,
        support_count,
        pair_count,
    )

    return CandidateRule(
        name=name,
        required_event_types=tuple(required_event_types),
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
        explanation=explanation,
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


# =============================================================================
# PUBLIC LEARNER
# =============================================================================

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
                required_event_types=shared_types,
                required_event_signatures=shared_signatures,
            )
        )

    meaningful_shared_types = tuple(
        event_type
        for event_type in shared_types
        if event_type != "object_preservation"
    )

    if (
        meaningful_shared_types
        and meaningful_shared_types != shared_types
    ):
        candidates.append(
            _build_candidate(
                stories,
                required_event_types=meaningful_shared_types,
                required_event_signatures=tuple(
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
                required_event_types=near_common_types,
                required_event_signatures=(),
            )
        )

    if not candidates:
        return LearnedRule(
            best_rule=None,
            candidates=(),
            pair_count=pair_count,
            warning=(
                "No transformation event was shared strongly enough "
                "across the train pairs to form a rule."
            ),
        )

    unique_candidates: dict[
        tuple[str, ...],
        CandidateRule,
    ] = {}

    for candidate in candidates:
        if candidate.support_count <= 0:
            continue

        key = candidate.required_event_types
        existing = unique_candidates.get(key)

        if (
            existing is None
            or candidate.confidence > existing.confidence
        ):
            unique_candidates[key] = candidate

    sorted_candidates = tuple(
        sorted(
            unique_candidates.values(),
            key=_candidate_sort_key,
            reverse=True,
        )
    )

    if not sorted_candidates:
        return LearnedRule(
            best_rule=None,
            candidates=(),
            pair_count=pair_count,
            warning=(
                "No supported candidate rule remained after filtering."
            ),
        )

    best_rule = sorted_candidates[0]

    warning = None

    if best_rule.support_ratio < 1.0:
        warning = (
            "The best rule is not supported by every train pair."
        )
    elif pair_count == 1:
        warning = (
            "Only one train pair is available, so the rule is descriptive "
            "but not yet strongly validated."
        )

    return LearnedRule(
        best_rule=best_rule,
        candidates=sorted_candidates,
        pair_count=pair_count,
        warning=warning,
    )


# =============================================================================
# FORMATTING
# =============================================================================

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

    lines = [
        title,
        "-" * len(title),
        f"Confidence : {candidate.confidence:.3f}",
        f"Consistency: {candidate.consistency:.3f}",
        f"Specificity: {candidate.specificity:.3f}",
        f"Simplicity : {candidate.simplicity:.3f}",
        f"Support    : {candidate.support_count}/{candidate.pair_count}",
        "",
        candidate.explanation,
    ]

    return "\n".join(lines)


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

        for index, candidate in enumerate(remaining, start=2):
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
        "It has not yet been executed against the train or test grids.",
    ])

    return "\n".join(lines)


# =============================================================================
# CONVENIENCE API
# =============================================================================

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


# =============================================================================
# SELF TEST
# =============================================================================

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

    output_1 = _shift_right(input_1, color=2, amount=3)
    output_2 = _shift_right(input_2, color=4, amount=2)

    learned = learn_rule_from_grid_pairs([
        (input_1, output_1),
        (input_2, output_2),
    ])

    print(format_learned_rule(learned))


if __name__ == "__main__":
    _self_test()