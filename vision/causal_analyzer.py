from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from vision.rule_learner import CandidateRule
from vision.story_generalizer import GeneralizedEvent, GeneralizedPairStory


# =============================================================================
# CAUSAL CATEGORIES
# =============================================================================

PRIMARY_ACTION_TYPES = {
    "object_translation",
    "object_rotation",
    "object_reflection",
    "object_recoloring",
    "object_resize",
    "object_shape_change",
    "object_creation",
    "object_deletion",
    "object_duplication",
    "object_merging",
    "object_splitting",
    "grid_resize",
    "canvas_change",
}

PRESERVATION_TYPES = {
    "object_preservation",
    "grid_preservation",
}

RELATIONSHIP_TYPES = {
    "relationship_creation",
    "relationship_deletion",
    "relationship_reversal",
    "relationship_change",
}

GEOMETRY_ACTION_TYPES = {
    "object_translation",
    "object_rotation",
    "object_reflection",
    "object_resize",
    "object_shape_change",
    "object_creation",
    "object_deletion",
    "object_duplication",
    "object_merging",
    "object_splitting",
    "grid_resize",
    "canvas_change",
}


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass(frozen=True)
class CausalEvent:
    event: GeneralizedEvent
    category: str
    reason: str

    @property
    def event_type(self) -> str:
        return self.event.event_type


@dataclass(frozen=True)
class PairCausalAnalysis:
    pair_index: int
    primary_actions: tuple[CausalEvent, ...]
    preserved_facts: tuple[CausalEvent, ...]
    goals_or_constraints: tuple[CausalEvent, ...]
    derived_consequences: tuple[CausalEvent, ...]
    unexplained_events: tuple[CausalEvent, ...]


@dataclass(frozen=True)
class CausalAnalysisReport:
    pair_analyses: tuple[PairCausalAnalysis, ...]
    common_relationship_event_types: tuple[str, ...]


# =============================================================================
# HELPERS
# =============================================================================

def _event_type_set(story: GeneralizedPairStory) -> set[str]:
    return {event.event_type for event in story.events}


def _common_relationship_types(
    stories: Sequence[GeneralizedPairStory],
) -> set[str]:
    if not stories:
        return set()

    relationship_sets = [
        _event_type_set(story) & RELATIONSHIP_TYPES
        for story in stories
    ]

    return set.intersection(*relationship_sets) if relationship_sets else set()


def _wrap(
    event: GeneralizedEvent,
    category: str,
    reason: str,
) -> CausalEvent:
    return CausalEvent(
        event=event,
        category=category,
        reason=reason,
    )


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_pair_causality(
    story: GeneralizedPairStory,
    candidate: CandidateRule,
    *,
    pair_index: int,
    common_relationship_event_types: set[str] | None = None,
) -> PairCausalAnalysis:
    common_relationship_event_types = (
        common_relationship_event_types or set()
    )

    required_types = set(candidate.required_event_types)
    story_types = _event_type_set(story)

    has_geometry_action = bool(
        story_types & GEOMETRY_ACTION_TYPES
    )

    primary_actions: list[CausalEvent] = []
    preserved_facts: list[CausalEvent] = []
    goals_or_constraints: list[CausalEvent] = []
    derived_consequences: list[CausalEvent] = []
    unexplained_events: list[CausalEvent] = []

    for event in story.events:
        event_type = event.event_type

        if event_type in PRESERVATION_TYPES:
            preserved_facts.append(
                _wrap(
                    event,
                    "preserved_fact",
                    "The event records what stayed unchanged.",
                )
            )
            continue

        if event_type in required_types:
            if event_type in RELATIONSHIP_TYPES:
                goals_or_constraints.append(
                    _wrap(
                        event,
                        "goal_or_constraint",
                        "The learned rule explicitly requires this relationship.",
                    )
                )
            else:
                primary_actions.append(
                    _wrap(
                        event,
                        "primary_action",
                        "The learned rule explicitly requires this transformation.",
                    )
                )
            continue

        if event_type in RELATIONSHIP_TYPES:
            if (
                event_type in common_relationship_event_types
                and not has_geometry_action
            ):
                goals_or_constraints.append(
                    _wrap(
                        event,
                        "goal_or_constraint",
                        "The same relationship change appears across every train pair.",
                    )
                )
            elif has_geometry_action:
                derived_consequences.append(
                    _wrap(
                        event,
                        "derived_consequence",
                        "A geometry-changing action can naturally produce this relationship change.",
                    )
                )
            else:
                unexplained_events.append(
                    _wrap(
                        event,
                        "unexplained_event",
                        "No primary geometry-changing action explains this relationship change.",
                    )
                )
            continue

        if event_type in PRIMARY_ACTION_TYPES:
            unexplained_events.append(
                _wrap(
                    event,
                    "unexplained_event",
                    "This independent transformation is not included in the learned rule.",
                )
            )
            continue

        unexplained_events.append(
            _wrap(
                event,
                "unexplained_event",
                "The event is neither required, preserved, nor causally explained.",
            )
        )

    return PairCausalAnalysis(
        pair_index=pair_index,
        primary_actions=tuple(primary_actions),
        preserved_facts=tuple(preserved_facts),
        goals_or_constraints=tuple(goals_or_constraints),
        derived_consequences=tuple(derived_consequences),
        unexplained_events=tuple(unexplained_events),
    )


def analyze_rule_causality(
    candidate: CandidateRule,
    stories: Sequence[GeneralizedPairStory],
) -> CausalAnalysisReport:
    common_relationship_types = _common_relationship_types(stories)

    pair_analyses = tuple(
        analyze_pair_causality(
            story,
            candidate,
            pair_index=index,
            common_relationship_event_types=common_relationship_types,
        )
        for index, story in enumerate(stories, start=1)
    )

    return CausalAnalysisReport(
        pair_analyses=pair_analyses,
        common_relationship_event_types=tuple(
            sorted(common_relationship_types)
        ),
    )


# =============================================================================
# FORMATTING
# =============================================================================

def _format_group(
    title: str,
    events: Sequence[CausalEvent],
) -> list[str]:
    lines = [title, "-" * 72]

    if not events:
        lines.append("  (none)")
        return lines

    for causal_event in events:
        lines.append(f"  - {causal_event.event_type}")
        lines.append(f"      {causal_event.reason}")

    return lines


def format_pair_causal_analysis(
    analysis: PairCausalAnalysis,
) -> str:
    lines = [
        f"TRAIN PAIR {analysis.pair_index}",
        "=" * 72,
    ]

    lines.extend(
        _format_group(
            "PRIMARY ACTIONS",
            analysis.primary_actions,
        )
    )
    lines.append("")

    lines.extend(
        _format_group(
            "PRESERVED FACTS",
            analysis.preserved_facts,
        )
    )
    lines.append("")

    lines.extend(
        _format_group(
            "GOALS / CONSTRAINTS",
            analysis.goals_or_constraints,
        )
    )
    lines.append("")

    lines.extend(
        _format_group(
            "DERIVED CONSEQUENCES",
            analysis.derived_consequences,
        )
    )
    lines.append("")

    lines.extend(
        _format_group(
            "UNEXPLAINED EVENTS",
            analysis.unexplained_events,
        )
    )

    return "\n".join(lines)


def format_causal_analysis(
    report: CausalAnalysisReport,
) -> str:
    lines = [
        "=" * 72,
        "CAUSAL ANALYSIS",
        "=" * 72,
    ]

    for analysis in report.pair_analyses:
        lines.append(format_pair_causal_analysis(analysis))
        lines.append("")

    return "\n".join(lines).rstrip()