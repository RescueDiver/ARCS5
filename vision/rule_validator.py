from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from vision.causal_analyzer import (
    PairCausalAnalysis,
    analyze_rule_causality,
)
from vision.rule_learner import (
    CandidateRule,
    LearnedRule,
    learn_rule,
)
from vision.story_generalizer import (
    GeneralizedEvent,
    GeneralizedPairStory,
    generalize_grid_pair,
)


# =============================================================================
# VALIDATION DATA TYPES
# =============================================================================

@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    expected: Any
    observed: Any
    explanation: str


@dataclass(frozen=True)
class PairValidationResult:
    pair_index: int
    passed: bool
    score: float
    checks: tuple[ValidationCheck, ...]
    primary_event_types: tuple[str, ...]
    preserved_event_types: tuple[str, ...]
    goal_event_types: tuple[str, ...]
    derived_event_types: tuple[str, ...]
    missing_event_types: tuple[str, ...]
    unexplained_event_types: tuple[str, ...]


@dataclass(frozen=True)
class RuleValidationReport:
    rule_name: str
    pair_count: int
    passed_pair_count: int
    failed_pair_count: int
    overall_score: float
    verdict: str
    pair_results: tuple[PairValidationResult, ...]
    warnings: tuple[str, ...]


# =============================================================================
# HELPERS
# =============================================================================

def _event_signature(event: GeneralizedEvent) -> tuple[Any, ...]:
    return event.signature(include_values=False)


def _story_event_types(story: GeneralizedPairStory) -> set[str]:
    return set(story.event_types)


def _story_signatures(
    story: GeneralizedPairStory,
) -> set[tuple[Any, ...]]:
    return {_event_signature(event) for event in story.events}


def _fact_check(
    fact: str,
    story: GeneralizedPairStory,
) -> ValidationCheck:
    event_types = _story_event_types(story)

    if fact == "moving_object.identity_preserved":
        passed = any(
            event.event_type == "object_translation"
            and dict(event.properties).get("preserves_identity") is True
            for event in story.events
        )
        return ValidationCheck(
            fact,
            passed,
            True,
            passed,
            (
                "Translated object preserves identity."
                if passed
                else "No translated object explicitly preserved identity."
            ),
        )

    if fact in {
        "moving_object.shape_preserved",
        "moving_object.color_preserved",
        "moving_object.size_preserved",
    }:
        passed = "object_translation" in event_types
        return ValidationCheck(
            fact,
            passed,
            True,
            passed,
            (
                "Translation preserves the moving object's basic identity."
                if passed
                else "No translated object was present."
            ),
        )

    if fact == "stationary_object.remains_unchanged":
        passed = "object_preservation" in event_types
        return ValidationCheck(
            fact,
            passed,
            True,
            passed,
            (
                "At least one matched object remained unchanged."
                if passed
                else "No unchanged matched object was found."
            ),
        )

    return ValidationCheck(
        fact,
        True,
        "descriptive fact",
        "not structurally testable yet",
        "This fact is retained but cannot yet be tested independently.",
    )


def _variable_fact_check(
    fact: str,
    story: GeneralizedPairStory,
) -> ValidationCheck:
    if fact.startswith("moving_object.") and "translation" in fact:
        passed = "object_translation" in story.event_types
        return ValidationCheck(
            fact,
            passed,
            "translation present",
            "translation present" if passed else "translation absent",
            "The learned rule permits this translation value to vary.",
        )

    return ValidationCheck(
        fact,
        True,
        "allowed variation",
        "not independently testable yet",
        "This variable fact is accepted but not yet measured.",
    )


def _types(events: Sequence[Any]) -> tuple[str, ...]:
    return tuple(sorted({event.event_type for event in events}))


# =============================================================================
# PAIR VALIDATION
# =============================================================================

def validate_candidate_against_story(
    candidate: CandidateRule,
    story: GeneralizedPairStory,
    causal_analysis: PairCausalAnalysis,
    *,
    pair_index: int,
) -> PairValidationResult:
    checks: list[ValidationCheck] = []
    observed_event_types = _story_event_types(story)
    required_event_types = set(candidate.required_event_types)

    missing = tuple(
        sorted(required_event_types - observed_event_types)
    )

    for event_type in candidate.required_event_types:
        passed = event_type in observed_event_types
        checks.append(
            ValidationCheck(
                f"required_event:{event_type}",
                passed,
                event_type,
                event_type if passed else "missing",
                (
                    f"Required event '{event_type}' was present."
                    if passed
                    else f"Required event '{event_type}' was absent."
                ),
            )
        )

    observed_signatures = _story_signatures(story)

    for signature in candidate.required_event_signatures:
        passed = signature in observed_signatures
        checks.append(
            ValidationCheck(
                "required_signature",
                passed,
                signature,
                signature if passed else "missing",
                (
                    "Required generalized event signature was present."
                    if passed
                    else "Required generalized event signature was absent."
                ),
            )
        )

    for fact in candidate.preserved_facts:
        checks.append(_fact_check(fact, story))

    for fact in candidate.variable_facts:
        checks.append(_variable_fact_check(fact, story))

    unexplained = _types(causal_analysis.unexplained_events)

    if unexplained:
        for event_type in unexplained:
            checks.append(
                ValidationCheck(
                    f"unexplained_event:{event_type}",
                    False,
                    "explained by rule or causal consequence",
                    event_type,
                    "This independent change is not explained by the learned rule.",
                )
            )

    score = (
        sum(check.passed for check in checks) / len(checks)
        if checks
        else 0.0
    )

    passed = (
        not missing
        and not unexplained
        and all(check.passed for check in checks)
    )

    return PairValidationResult(
        pair_index=pair_index,
        passed=passed,
        score=score,
        checks=tuple(checks),
        primary_event_types=_types(causal_analysis.primary_actions),
        preserved_event_types=_types(causal_analysis.preserved_facts),
        goal_event_types=_types(causal_analysis.goals_or_constraints),
        derived_event_types=_types(causal_analysis.derived_consequences),
        missing_event_types=missing,
        unexplained_event_types=unexplained,
    )


# =============================================================================
# RULE VALIDATION
# =============================================================================

def validate_candidate_rule(
    candidate: CandidateRule,
    stories: Sequence[GeneralizedPairStory],
) -> RuleValidationReport:
    causal_report = analyze_rule_causality(candidate, stories)

    pair_results = tuple(
        validate_candidate_against_story(
            candidate,
            story,
            causal_analysis,
            pair_index=index,
        )
        for index, (story, causal_analysis) in enumerate(
            zip(stories, causal_report.pair_analyses),
            start=1,
        )
    )

    passed_count = sum(result.passed for result in pair_results)
    failed_count = len(pair_results) - passed_count
    overall_score = (
        sum(result.score for result in pair_results) / len(pair_results)
        if pair_results
        else 0.0
    )

    warnings: list[str] = []

    if not pair_results:
        verdict = "REJECT"
        warnings.append("No generalized train stories were provided.")
    elif passed_count == len(pair_results):
        verdict = "PASS"
    elif overall_score >= 0.80:
        verdict = "PARTIAL"
        warnings.append(
            "The rule explains most evidence but at least one train pair fails."
        )
    else:
        verdict = "REJECT"
        warnings.append(
            "The rule does not explain the training evidence consistently."
        )

    for result in pair_results:
        if result.unexplained_event_types:
            warnings.append(
                f"Pair {result.pair_index} has unexplained events: "
                + ", ".join(result.unexplained_event_types)
            )

    return RuleValidationReport(
        rule_name=candidate.name,
        pair_count=len(pair_results),
        passed_pair_count=passed_count,
        failed_pair_count=failed_count,
        overall_score=overall_score,
        verdict=verdict,
        pair_results=pair_results,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def validate_learned_rule(
    learned: LearnedRule,
    stories: Sequence[GeneralizedPairStory],
) -> RuleValidationReport:
    if learned.best_rule is None:
        return RuleValidationReport(
            rule_name="No learned rule",
            pair_count=len(stories),
            passed_pair_count=0,
            failed_pair_count=len(stories),
            overall_score=0.0,
            verdict="REJECT",
            pair_results=(),
            warnings=(
                learned.warning or "The learner did not produce a best rule.",
            ),
        )

    return validate_candidate_rule(learned.best_rule, stories)


# =============================================================================
# FORMATTING
# =============================================================================

def _add_event_group(
    lines: list[str],
    title: str,
    event_types: tuple[str, ...],
    mark: str,
) -> None:
    lines.append(title + ":")

    if event_types:
        for event_type in event_types:
            lines.append(f"  {mark} {event_type}")
    else:
        lines.append("  (none)")

    lines.append("")


def format_pair_validation(
    result: PairValidationResult,
) -> str:
    lines = [
        f"TRAIN PAIR {result.pair_index}: "
        f"{'PASS' if result.passed else 'FAIL'}",
        "-" * 72,
        f"Validation score: {result.score:.3f}",
        "",
    ]

    _add_event_group(
        lines,
        "Primary actions",
        result.primary_event_types,
        "✓",
    )
    _add_event_group(
        lines,
        "Preserved facts",
        result.preserved_event_types,
        "✓",
    )
    _add_event_group(
        lines,
        "Goals / constraints",
        result.goal_event_types,
        "◎",
    )
    _add_event_group(
        lines,
        "Derived consequences",
        result.derived_event_types,
        "↳",
    )

    if result.missing_event_types:
        _add_event_group(
            lines,
            "Missing required events",
            result.missing_event_types,
            "✗",
        )

    _add_event_group(
        lines,
        "Unexplained events",
        result.unexplained_event_types,
        "!",
    )

    lines.append("Checks:")

    for check in result.checks:
        mark = "✓" if check.passed else "✗"
        lines.append(f"  {mark} {check.name}")

        if not check.passed:
            lines.append(f"      expected: {check.expected}")
            lines.append(f"      observed: {check.observed}")

    return "\n".join(lines)


def format_rule_validation(
    report: RuleValidationReport,
) -> str:
    lines = [
        "=" * 72,
        "RULE VALIDATION",
        "=" * 72,
        f"Rule         : {report.rule_name}",
        f"Verdict      : {report.verdict}",
        f"Overall score: {report.overall_score:.3f}",
        (
            f"Train pairs  : "
            f"{report.passed_pair_count}/{report.pair_count} passed"
        ),
        "",
    ]

    for result in report.pair_results:
        lines.append(format_pair_validation(result))
        lines.append("")

    if report.warnings:
        lines.extend(["WARNINGS", "-" * 72])
        for warning in report.warnings:
            lines.append(f"  ! {warning}")
        lines.append("")

    if report.verdict == "PASS":
        lines.append(
            "The learned rule explains every train pair after causal "
            "consequences are separated from independent changes."
        )
    elif report.verdict == "PARTIAL":
        lines.append(
            "The rule is not safe to execute because at least one train "
            "pair still contains missing or unexplained evidence."
        )
    else:
        lines.append(
            "The rule must be rejected or refined before execution."
        )

    return "\n".join(lines)


# =============================================================================
# CONVENIENCE API
# =============================================================================

def learn_and_validate_rule(
    stories: Sequence[GeneralizedPairStory],
) -> tuple[LearnedRule, RuleValidationReport]:
    learned = learn_rule(stories)
    return learned, validate_learned_rule(learned, stories)


def learn_and_validate_grid_pairs(
    train_pairs: Sequence[
        tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]
    ],
    *,
    connectivity: int = 4,
    include_preservations: bool = True,
) -> tuple[LearnedRule, RuleValidationReport]:
    stories = tuple(
        generalize_grid_pair(
            input_grid,
            output_grid,
            connectivity=connectivity,
            include_preservations=include_preservations,
        )
        for input_grid, output_grid in train_pairs
    )

    return learn_and_validate_rule(stories)


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

    _, report = learn_and_validate_grid_pairs([
        (input_1, output_1),
        (input_2, output_2),
    ])

    print(format_rule_validation(report))


if __name__ == "__main__":
    _self_test()
