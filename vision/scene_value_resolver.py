from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from vision.rule_planner import MOVE_OBJECT, PlanArgument, PlanStep, RulePlan
from vision.scene_graph import SceneGraph, SceneGraphNode, build_scene_graph
from vision.story_generalizer import GeneralizedPairStory
from vision.value_inference import ArgumentInference, InferenceCandidate, ValueInferenceReport


@dataclass(frozen=True)
class SceneRoleBindings:
    """Concrete test-scene object bindings supplied by object selection."""

    transformed_object_id: int
    anchor_object_id: int


@dataclass(frozen=True)
class ResolvedSceneValue:
    step_number: int
    operation: str
    argument_name: str
    value: Any
    confidence: float
    resolver: str
    explanation: str


@dataclass(frozen=True)
class SceneValueResolutionReport:
    plan_ready_before: bool
    plan_ready_after: bool
    resolved_plan: RulePlan
    resolved_values: tuple[ResolvedSceneValue, ...]
    unresolved_values: tuple[str, ...]
    warnings: tuple[str, ...]


DIRECTIONAL_RELATIONSHIPS = {"left_of", "right_of", "above", "below"}


def _properties(event: Any) -> dict[str, Any]:
    raw = getattr(event, "properties", ())
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        return dict(raw)
    except (TypeError, ValueError):
        return {}


def infer_relationship_goal(
    stories: Sequence[GeneralizedPairStory],
) -> str | None:
    """Return one directional output relationship shared by all train pairs."""

    goals: list[str] = []

    for story in stories:
        pair_goals: list[str] = []

        for event in story.events:
            props = _properties(event)

            if event.event_type == "relationship_reversal":
                relationship = props.get("new_relationship")
            elif event.event_type == "relationship_creation":
                relationship = props.get("relationship")
            else:
                relationship = None

            if relationship in DIRECTIONAL_RELATIONSHIPS:
                pair_goals.append(str(relationship))

        unique = tuple(dict.fromkeys(pair_goals))
        if len(unique) != 1:
            return None
        goals.append(unique[0])

    if not goals or not all(goal == goals[0] for goal in goals):
        return None

    return goals[0]


def _required_axis_delta(
    mover: SceneGraphNode,
    anchor: SceneGraphNode,
    relationship: str,
) -> tuple[str, int]:
    """Return the smallest strict-separation translation for the goal."""

    if relationship == "right_of":
        return "column_delta", anchor.bbox_right - mover.bbox_left + 1
    if relationship == "left_of":
        return "column_delta", anchor.bbox_left - mover.bbox_right - 1
    if relationship == "below":
        return "row_delta", anchor.bbox_bottom - mover.bbox_top + 1
    if relationship == "above":
        return "row_delta", anchor.bbox_top - mover.bbox_bottom - 1

    raise ValueError(f"Unsupported relationship: {relationship!r}")


def _relationship_holds(
    mover: SceneGraphNode,
    anchor: SceneGraphNode,
    relationship: str,
    *,
    row_delta: int,
    column_delta: int,
) -> bool:
    top = mover.bbox_top + row_delta
    bottom = mover.bbox_bottom + row_delta
    left = mover.bbox_left + column_delta
    right = mover.bbox_right + column_delta

    if relationship == "right_of":
        return left > anchor.bbox_right
    if relationship == "left_of":
        return right < anchor.bbox_left
    if relationship == "below":
        return top > anchor.bbox_bottom
    if relationship == "above":
        return bottom < anchor.bbox_top
    return False


def _translated_cells(
    node: SceneGraphNode,
    *,
    row_delta: int,
    column_delta: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (row + row_delta, column + column_delta)
        for row, column in node.cells
    )


def validate_translation(
    graph: SceneGraph,
    mover: SceneGraphNode,
    anchor: SceneGraphNode,
    relationship: str,
    *,
    row_delta: int,
    column_delta: int,
) -> tuple[bool, str]:
    moved_cells = _translated_cells(
        mover,
        row_delta=row_delta,
        column_delta=column_delta,
    )

    if not all(
        0 <= row < graph.height and 0 <= column < graph.width
        for row, column in moved_cells
    ):
        return False, "The moved object would leave the grid."

    occupied = {
        cell
        for node in graph.nodes
        if node.object_id != mover.object_id
        and node.role.value not in {"background", "void"}
        for cell in node.cells
    }

    if any(cell in occupied for cell in moved_cells):
        return False, "The moved object would overlap another object."

    if not _relationship_holds(
        mover,
        anchor,
        relationship,
        row_delta=row_delta,
        column_delta=column_delta,
    ):
        return False, f"The move does not satisfy {relationship!r}."

    return True, "The translation is geometrically valid."


def _argument_inference(
    report: ValueInferenceReport,
    step_number: int,
    argument_name: str,
) -> ArgumentInference | None:
    return next(
        (
            inference
            for inference in report.argument_inferences
            if inference.step_number == step_number
            and inference.argument_name == argument_name
        ),
        None,
    )


def _best_candidate(
    inference: ArgumentInference | None,
) -> InferenceCandidate | None:
    if inference is None or not inference.candidates:
        return None
    return inference.candidates[0]


def _candidate_sign(candidate: InferenceCandidate | None) -> int | None:
    if candidate is None or not isinstance(candidate.inferred_value, Mapping):
        return None

    sign = candidate.inferred_value.get("sign")
    return int(sign) if sign in {-1, 0, 1} else None


def _fixed_integer(step: PlanStep, name: str) -> int | None:
    for argument in step.arguments:
        if (
            argument.name == name
            and argument.resolved
            and isinstance(argument.value, int)
            and not isinstance(argument.value, bool)
        ):
            return argument.value
    return None


def infer_distance_from_test_scene(
    graph: SceneGraph,
    bindings: SceneRoleBindings,
    *,
    relationship_goal: str,
    row_direction_sign: int | None = None,
    column_direction_sign: int | None = None,
    fixed_row_delta: int | None = None,
    fixed_column_delta: int | None = None,
) -> ResolvedSceneValue:
    """
    Resolve one unknown movement distance after mover and anchor are bound.

    Object selection is intentionally outside this module.
    """

    mover = graph.node(bindings.transformed_object_id)
    anchor = graph.node(bindings.anchor_object_id)

    if mover is None:
        raise ValueError("The transformed object does not exist in the test scene.")
    if anchor is None:
        raise ValueError("The anchor object does not exist in the test scene.")

    axis_name, required_delta = _required_axis_delta(
        mover,
        anchor,
        relationship_goal,
    )

    row_delta = fixed_row_delta if fixed_row_delta is not None else 0
    column_delta = fixed_column_delta if fixed_column_delta is not None else 0

    if axis_name == "row_delta":
        row_delta = required_delta
        learned_sign = row_direction_sign
    else:
        column_delta = required_delta
        learned_sign = column_direction_sign

    actual_sign = 1 if required_delta > 0 else -1 if required_delta < 0 else 0

    if learned_sign is not None and actual_sign != learned_sign:
        raise ValueError(
            f"The scene requires {axis_name}={required_delta}, "
            f"but training requires sign {learned_sign}."
        )

    valid, reason = validate_translation(
        graph,
        mover,
        anchor,
        relationship_goal,
        row_delta=row_delta,
        column_delta=column_delta,
    )
    if not valid:
        raise ValueError(reason)

    return ResolvedSceneValue(
        step_number=-1,
        operation=MOVE_OBJECT,
        argument_name=axis_name,
        value=required_delta,
        confidence=1.0,
        resolver="infer_distance_from_test_scene",
        explanation=(
            f"Object {mover.object_id} must use {axis_name}={required_delta} "
            f"to become {relationship_goal} Object {anchor.object_id}."
        ),
    )


def _resolved_argument(
    argument: PlanArgument,
    result: ResolvedSceneValue,
) -> PlanArgument:
    return replace(
        argument,
        value=result.value,
        source=(
            f"test-scene resolver: {result.resolver}; "
            f"{result.explanation}"
        ),
        resolved=True,
    )


def resolve_test_scene_values(
    inference_report: ValueInferenceReport,
    training_stories: Sequence[GeneralizedPairStory],
    test_grid: Sequence[Sequence[int]],
    bindings: SceneRoleBindings,
    *,
    connectivity: int = 4,
    preferred_void_colors: Iterable[int] = (0,),
    background_hint: int | None = None,
) -> SceneValueResolutionReport:
    """Resolve current MOVE_OBJECT test-specific values and rebuild the plan."""

    plan = inference_report.resolved_plan
    graph = build_scene_graph(
        test_grid,
        connectivity=connectivity,
        preferred_void_colors=preferred_void_colors,
        background_hint=background_hint,
    )
    relationship_goal = infer_relationship_goal(training_stories)

    warnings: list[str] = []
    resolved_values: list[ResolvedSceneValue] = []
    new_steps: list[PlanStep] = []

    if relationship_goal is None:
        warnings.append(
            "No single directional relationship goal was shared by all train pairs."
        )

    for step in plan.steps:
        if step.operation != MOVE_OBJECT or relationship_goal is None:
            new_steps.append(step)
            continue

        row_candidate = _best_candidate(
            _argument_inference(inference_report, step.step_number, "row_delta")
        )
        column_candidate = _best_candidate(
            _argument_inference(inference_report, step.step_number, "column_delta")
        )

        try:
            result = infer_distance_from_test_scene(
                graph,
                bindings,
                relationship_goal=relationship_goal,
                row_direction_sign=_candidate_sign(row_candidate),
                column_direction_sign=_candidate_sign(column_candidate),
                fixed_row_delta=_fixed_integer(step, "row_delta"),
                fixed_column_delta=_fixed_integer(step, "column_delta"),
            )
            result = replace(result, step_number=step.step_number)
        except ValueError as error:
            warnings.append(f"Step {step.step_number}: {error}")
            new_steps.append(step)
            continue

        new_arguments = tuple(
            _resolved_argument(argument, result)
            if argument.name == result.argument_name and not argument.resolved
            else argument
            for argument in step.arguments
        )

        resolved_values.append(result)
        new_steps.append(
            replace(
                step,
                arguments=new_arguments,
                executable=all(argument.resolved for argument in new_arguments),
            )
        )

    unresolved = tuple(
        dict.fromkeys(
            f"step {step.step_number}: {argument.name}"
            for step in new_steps
            for argument in step.arguments
            if not argument.resolved
        )
    )

    ready = (
        plan.validation_verdict == "PASS"
        and not plan.unsupported_events
        and not unresolved
    )

    resolved_plan = replace(
        plan,
        steps=tuple(new_steps),
        unresolved_values=unresolved,
        ready_for_execution=ready,
        warnings=tuple(dict.fromkeys(tuple(plan.warnings) + tuple(warnings))),
    )

    return SceneValueResolutionReport(
        plan_ready_before=plan.ready_for_execution,
        plan_ready_after=ready,
        resolved_plan=resolved_plan,
        resolved_values=tuple(resolved_values),
        unresolved_values=unresolved,
        warnings=tuple(warnings),
    )


def format_scene_value_resolution(
    report: SceneValueResolutionReport,
) -> str:
    lines = [
        "=" * 72,
        "TEST-SCENE VALUE RESOLUTION",
        "=" * 72,
        "Plan ready before : " + ("YES" if report.plan_ready_before else "NO"),
        "Plan ready after  : " + ("YES" if report.plan_ready_after else "NO"),
        "",
        "RESOLVED VALUES",
        "-" * 72,
    ]

    if report.resolved_values:
        for result in report.resolved_values:
            lines.extend([
                f"STEP {result.step_number} / {result.operation} / {result.argument_name}",
                f"  Value      : {result.value!r}",
                f"  Confidence : {result.confidence:.3f}",
                f"  Resolver   : {result.resolver}",
                f"  Explanation: {result.explanation}",
                "",
            ])
    else:
        lines.extend(["  (none)", ""])

    if report.unresolved_values:
        lines.extend(["STILL UNRESOLVED", "-" * 72])
        lines.extend(f"  ? {value}" for value in report.unresolved_values)
        lines.append("")

    if report.warnings:
        lines.extend(["WARNINGS", "-" * 72])
        lines.extend(f"  ! {warning}" for warning in report.warnings)

    return "\n".join(lines)


def _self_test() -> None:
    test_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 0],
        [0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    graph = build_scene_graph(test_grid, connectivity=4)
    mover = next(node for node in graph.nodes if node.color == 2 and node.cell_count == 4)
    anchor = next(node for node in graph.nodes if node.color == 3 and node.cell_count == 1)

    result = infer_distance_from_test_scene(
        graph,
        SceneRoleBindings(
            transformed_object_id=mover.object_id,
            anchor_object_id=anchor.object_id,
        ),
        relationship_goal="right_of",
        row_direction_sign=0,
        column_direction_sign=1,
        fixed_row_delta=0,
    )

    print("=" * 72)
    print("SCENE VALUE RESOLVER SELF TEST")
    print("=" * 72)
    print(f"Argument   : {result.argument_name}")
    print(f"Value      : {result.value}")
    print(f"Confidence : {result.confidence:.3f}")
    print(f"Explanation: {result.explanation}")


if __name__ == "__main__":
    _self_test()