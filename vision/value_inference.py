from __future__ import annotations

from dataclasses import dataclass, replace
from math import gcd
from typing import Any, Iterable, Mapping, Sequence

from vision.rule_learner import LearnedRule
from vision.rule_planner import PlanArgument, PlanStep, RulePlan, build_rule_plan
from vision.rule_validator import RuleValidationReport, validate_learned_rule
from vision.story_generalizer import GeneralizedEvent, GeneralizedPairStory, generalize_grid_pair


@dataclass(frozen=True)
class ObservedValue:
    pair_index: int
    argument_name: str
    value: Any
    source_event_type: str
    source_properties: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class InferenceCandidate:
    argument_name: str
    strategy: str
    inferred_value: Any
    confidence: float
    support_count: int
    pair_count: int
    explanation: str
    test_resolver: str | None = None


@dataclass(frozen=True)
class ArgumentInference:
    step_number: int
    operation: str
    argument_name: str
    candidates: tuple[InferenceCandidate, ...]
    selected: InferenceCandidate | None
    resolved: bool
    warning: str | None = None


@dataclass(frozen=True)
class ValueInferenceReport:
    rule_name: str
    plan_ready_before: bool
    plan_ready_after: bool
    resolved_plan: RulePlan
    argument_inferences: tuple[ArgumentInference, ...]
    warnings: tuple[str, ...]


ROW_KEYS = ("row_delta", "delta_row", "dr", "dy", "vertical_delta", "row_shift")
COLUMN_KEYS = ("column_delta", "col_delta", "delta_column", "delta_col", "dc", "dx", "horizontal_delta", "column_shift", "col_shift")
DISTANCE_KEYS = ("distance", "translation_distance", "magnitude", "steps")
ROTATION_KEYS = ("rotation", "rotation_degrees", "degrees", "turns", "quarter_turns")
AXIS_KEYS = ("axis", "reflection_axis", "mirror_axis")
COLOR_KEYS = ("target_color", "to_color", "new_color", "output_color")
SIZE_KEYS = ("scale_or_dimensions", "target_size", "output_size", "dimensions", "scale")
PLACEMENT_KEYS = ("placement", "target_position", "output_position", "destination")

OPERATION_EVENT_TYPES = {
    "MOVE_OBJECT": {"object_translation"},
    "ROTATE_OBJECT": {"object_rotation"},
    "REFLECT_OBJECT": {"object_reflection"},
    "RECOLOR_OBJECT": {"object_recoloring"},
    "RESIZE_OBJECT": {"object_resize"},
    "CHANGE_SHAPE": {"object_shape_change"},
    "CREATE_OBJECT": {"object_creation"},
    "DELETE_OBJECT": {"object_deletion"},
    "DUPLICATE_OBJECT": {"object_duplication"},
    "MERGE_OBJECTS": {"object_merging"},
    "SPLIT_OBJECT": {"object_splitting"},
    "RESIZE_GRID": {"grid_resize"},
    "CHANGE_CANVAS": {"canvas_change"},
    "SATISFY_RELATIONSHIP": {"relationship_creation", "relationship_deletion", "relationship_reversal", "relationship_change"},
}


def _properties(event: GeneralizedEvent) -> dict[str, Any]:
    raw = getattr(event, "properties", ())
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        return dict(raw)
    except (TypeError, ValueError):
        return {}


def _first_present(properties: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in properties:
            return properties[key]
    return None


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _extract_translation(event: GeneralizedEvent) -> tuple[int | None, int | None]:
    properties = _properties(event)
    row_delta = _to_int(_first_present(properties, ROW_KEYS))
    column_delta = _to_int(_first_present(properties, COLUMN_KEYS))
    vector = properties.get("translation_vector")
    if isinstance(vector, (tuple, list)) and len(vector) == 2:
        if row_delta is None:
            row_delta = _to_int(vector[0])
        if column_delta is None:
            column_delta = _to_int(vector[1])

    direction = properties.get("direction")
    distance = _to_int(_first_present(properties, DISTANCE_KEYS))
    if isinstance(direction, str) and distance is not None:
        key = direction.strip().lower().replace("-", "_")
        vectors = {
            "up": (-distance, 0), "down": (distance, 0),
            "left": (0, -distance), "right": (0, distance),
            "up_left": (-distance, -distance), "upper_left": (-distance, -distance),
            "up_right": (-distance, distance), "upper_right": (-distance, distance),
            "down_left": (distance, -distance), "lower_left": (distance, -distance),
            "down_right": (distance, distance), "lower_right": (distance, distance),
        }
        inferred = vectors.get(key)
        if inferred:
            row_delta = inferred[0] if row_delta is None else row_delta
            column_delta = inferred[1] if column_delta is None else column_delta
    return row_delta, column_delta


def _extract_argument_value(event: GeneralizedEvent, argument_name: str) -> Any:
    properties = _properties(event)
    if argument_name == "row_delta":
        return _extract_translation(event)[0]
    if argument_name == "column_delta":
        return _extract_translation(event)[1]
    if argument_name == "rotation":
        return _first_present(properties, ROTATION_KEYS)
    if argument_name == "axis":
        return _first_present(properties, AXIS_KEYS)
    if argument_name == "target_color":
        return _first_present(properties, COLOR_KEYS)
    if argument_name == "scale_or_dimensions":
        return _first_present(properties, SIZE_KEYS)
    if argument_name in {"placement", "placement_or_adjustment"}:
        return _first_present(properties, PLACEMENT_KEYS)
    if argument_name == "shape":
        return properties.get("target_shape") or properties.get("shape")
    return properties.get(argument_name)


def collect_argument_observations(stories: Sequence[GeneralizedPairStory], *, operation: str, argument_name: str) -> tuple[ObservedValue, ...]:
    allowed = OPERATION_EVENT_TYPES.get(operation, set())
    observations: list[ObservedValue] = []
    for pair_index, story in enumerate(stories, start=1):
        for event in story.events:
            if allowed and event.event_type not in allowed:
                continue
            value = _extract_argument_value(event, argument_name)
            if value is None:
                continue
            observations.append(ObservedValue(
                pair_index=pair_index,
                argument_name=argument_name,
                value=value,
                source_event_type=event.event_type,
                source_properties=tuple(sorted(_properties(event).items(), key=lambda item: item[0])),
            ))
    return tuple(observations)


def _all_equal(values: Sequence[Any]) -> bool:
    return bool(values) and all(value == values[0] for value in values[1:])


def _sign(value: int) -> int:
    return -1 if value < 0 else (1 if value > 0 else 0)


def _numeric_candidates(argument_name: str, values: Sequence[int], pair_count: int) -> list[InferenceCandidate]:
    candidates: list[InferenceCandidate] = []
    signs = [_sign(value) for value in values]
    if _all_equal(signs):
        candidates.append(InferenceCandidate(
            argument_name, "fixed_direction_variable_distance",
            {"sign": signs[0], "magnitude": None}, 0.84, len(values), pair_count,
            f"The {argument_name} direction is consistent, but its magnitude varies.",
            "infer_distance_from_test_scene",
        ))
    absolute_values = [abs(value) for value in values]
    if _all_equal(absolute_values):
        candidates.append(InferenceCandidate(
            argument_name, "fixed_magnitude_variable_direction",
            {"sign": None, "magnitude": absolute_values[0]}, 0.78, len(values), pair_count,
            f"The magnitude of {argument_name} is always {absolute_values[0]}, but direction varies.",
            "infer_direction_from_test_scene",
        ))
    nonzero = [abs(value) for value in values if value != 0]
    if nonzero:
        common_step = nonzero[0]
        for value in nonzero[1:]:
            common_step = gcd(common_step, value)
        if common_step > 1:
            candidates.append(InferenceCandidate(
                argument_name, "multiple_of_common_step",
                {"step": common_step, "multiplier": None}, 0.58, len(values), pair_count,
                f"Observed values are multiples of {common_step}; the multiplier must come from the test scene.",
                "infer_multiplier_from_test_scene",
            ))
    return candidates


def infer_argument_candidates(argument_name: str, observations: Sequence[ObservedValue], *, pair_count: int) -> tuple[InferenceCandidate, ...]:
    values = [observation.value for observation in observations]
    if not values:
        return ()
    if _all_equal(values):
        candidates = [InferenceCandidate(
            argument_name, "fixed_value", values[0], 1.0, len(values), pair_count,
            f"Every observed train pair uses the same {argument_name}: {values[0]!r}.",
        )]
    elif all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        candidates = _numeric_candidates(argument_name, values, pair_count)
    else:
        counts: dict[str, tuple[Any, int]] = {}
        for value in values:
            key = repr(value)
            previous = counts.get(key)
            counts[key] = (value, 1 if previous is None else previous[1] + 1)
        mode_value, support = max(counts.values(), key=lambda item: item[1])
        candidates = [InferenceCandidate(
            argument_name, "most_common_value", mode_value,
            support / len(values), support, pair_count,
            f"The most common observed {argument_name} is {mode_value!r}, supported by {support}/{len(values)} observations.",
        )]
    candidates.sort(key=lambda candidate: (candidate.confidence, candidate.support_count, candidate.strategy == "fixed_value"), reverse=True)
    return tuple(candidates)


def select_candidate(candidates: Sequence[InferenceCandidate], *, minimum_confidence: float = 0.95) -> InferenceCandidate | None:
    if not candidates:
        return None
    best = candidates[0]
    if best.confidence >= minimum_confidence and best.test_resolver is None:
        return best
    return None


def infer_plan_values(plan: RulePlan, stories: Sequence[GeneralizedPairStory], *, minimum_confidence: float = 0.95) -> ValueInferenceReport:
    reports: list[ArgumentInference] = []
    resolved_steps: list[PlanStep] = []
    warnings: list[str] = []

    for step in plan.steps:
        new_arguments: list[PlanArgument] = []
        for argument in step.arguments:
            if argument.resolved:
                new_arguments.append(argument)
                continue
            observations = collect_argument_observations(stories, operation=step.operation, argument_name=argument.name)
            candidates = infer_argument_candidates(argument.name, observations, pair_count=len(stories))
            selected = select_candidate(candidates, minimum_confidence=minimum_confidence)
            warning = None
            if not observations:
                warning = "No concrete training values were retained in the generalized events."
            elif selected is None and candidates:
                warning = f"Best strategy '{candidates[0].strategy}' requires test-scene reasoning or has insufficient confidence."
            elif selected is None:
                warning = "No viable inference candidate was found."
            reports.append(ArgumentInference(step.step_number, step.operation, argument.name, candidates, selected, selected is not None, warning))
            if selected is None:
                new_arguments.append(argument)
            else:
                new_arguments.append(replace(argument, value=selected.inferred_value, source=f"value inference: {selected.strategy}; {selected.explanation}", resolved=True))
        resolved_steps.append(replace(step, arguments=tuple(new_arguments), executable=all(argument.resolved for argument in new_arguments)))

    unresolved = tuple(dict.fromkeys(
        f"step {step.step_number}: {argument.name}"
        for step in resolved_steps
        for argument in step.arguments
        if not argument.resolved
    ))
    if unresolved:
        warnings.append("Some values remain unresolved because training evidence does not uniquely determine a concrete test value.")
    ready = plan.validation_verdict == "PASS" and not plan.unsupported_events and not unresolved
    resolved_plan = replace(
        plan,
        steps=tuple(resolved_steps),
        unresolved_values=unresolved,
        ready_for_execution=ready,
        warnings=tuple(dict.fromkeys(tuple(plan.warnings) + tuple(warnings))),
    )
    return ValueInferenceReport(plan.rule_name, plan.ready_for_execution, ready, resolved_plan, tuple(reports), tuple(warnings))


def learn_validate_plan_and_infer(learned: LearnedRule, stories: Sequence[GeneralizedPairStory], *, minimum_confidence: float = 0.95) -> tuple[RuleValidationReport, RulePlan, ValueInferenceReport]:
    validation = validate_learned_rule(learned, stories)
    plan = build_rule_plan(learned, validation, stories)
    return validation, plan, infer_plan_values(plan, stories, minimum_confidence=minimum_confidence)


def format_inference_candidate(candidate: InferenceCandidate) -> str:
    resolver = candidate.test_resolver or "none required"
    return "\n".join([
        f"    Strategy   : {candidate.strategy}",
        f"    Value      : {candidate.inferred_value!r}",
        f"    Confidence : {candidate.confidence:.3f}",
        f"    Support    : {candidate.support_count}/{candidate.pair_count}",
        f"    Resolver   : {resolver}",
        f"    Explanation: {candidate.explanation}",
    ])


def format_value_inference(report: ValueInferenceReport) -> str:
    lines = [
        "=" * 72, "VALUE INFERENCE", "=" * 72,
        f"Rule              : {report.rule_name}",
        "Plan ready before : " + ("YES" if report.plan_ready_before else "NO"),
        "Plan ready after  : " + ("YES" if report.plan_ready_after else "NO"), "",
    ]
    for inference in report.argument_inferences:
        lines.extend([
            f"STEP {inference.step_number} / {inference.operation} / {inference.argument_name}",
            "-" * 72,
            "Resolved: " + ("YES" if inference.resolved else "NO"),
        ])
        if inference.selected:
            lines.extend(["", "SELECTED", format_inference_candidate(inference.selected)])
        lines.extend(["", "CANDIDATES"])
        if inference.candidates:
            for index, candidate in enumerate(inference.candidates, start=1):
                lines.extend([f"  Candidate {index}", format_inference_candidate(candidate)])
        else:
            lines.append("  (none)")
        if inference.warning:
            lines.extend(["", f"Warning: {inference.warning}"])
        lines.append("")
    if report.resolved_plan.unresolved_values:
        lines.extend(["STILL UNRESOLVED", "-" * 72])
        lines.extend(f"  ? {value}" for value in report.resolved_plan.unresolved_values)
        lines.append("")
    if report.warnings:
        lines.extend(["WARNINGS", "-" * 72])
        lines.extend(f"  ! {warning}" for warning in report.warnings)
        lines.append("")
    lines.append(
        "Every required plan value is concrete. The resolved plan may be passed to a primitive executor."
        if report.plan_ready_after
        else "The plan remains non-executable until a test-scene resolver supplies the remaining values."
    )
    return "\n".join(lines)


def _shift(grid: list[list[int]], color: int, row_delta: int, column_delta: int) -> list[list[int]]:
    output = [row[:] for row in grid]
    cells: list[tuple[int, int]] = []
    for row_index, row in enumerate(grid):
        for column_index, value in enumerate(row):
            if value == color:
                cells.append((row_index, column_index))
                output[row_index][column_index] = 0
    for row_index, column_index in cells:
        output[row_index + row_delta][column_index + column_delta] = color
    return output


def _self_test() -> None:
    from vision.rule_learner import learn_rule

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
    output_1 = _shift(input_1, 2, 0, 3)
    output_2 = _shift(input_2, 4, 0, 2)
    stories = tuple(
        generalize_grid_pair(input_grid, output_grid, connectivity=4, include_preservations=True)
        for input_grid, output_grid in [(input_1, output_1), (input_2, output_2)]
    )
    learned = learn_rule(stories)
    validation = validate_learned_rule(learned, stories)
    plan = build_rule_plan(learned, validation, stories)
    report = infer_plan_values(plan, stories)
    print(format_value_inference(report))


if __name__ == "__main__":
    _self_test()