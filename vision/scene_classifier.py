from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from vision.objects import ObjectDetectionResult, VisualObject, detect_objects


# =============================================================================
# SCENE ROLES
# =============================================================================

class SceneRole(str, Enum):
    BACKGROUND = "background"
    FOREGROUND = "foreground"
    VOID = "void"
    DIVIDER = "divider"
    FRAME = "frame"
    DECORATION = "decoration"
    UNCERTAIN = "uncertain"


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass(frozen=True)
class SceneElement:
    object_id: int
    role: SceneRole
    confidence: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SceneClassificationResult:
    height: int
    width: int
    connectivity: int
    elements: tuple[SceneElement, ...]

    def element_for_object(self, object_id: int) -> SceneElement | None:
        for element in self.elements:
            if element.object_id == object_id:
                return element
        return None

    def objects_with_role(self, role: SceneRole) -> tuple[int, ...]:
        return tuple(
            element.object_id
            for element in self.elements
            if element.role == role
        )

    @property
    def background_object_ids(self) -> tuple[int, ...]:
        return self.objects_with_role(SceneRole.BACKGROUND)

    @property
    def foreground_object_ids(self) -> tuple[int, ...]:
        return self.objects_with_role(SceneRole.FOREGROUND)

    @property
    def void_object_ids(self) -> tuple[int, ...]:
        return self.objects_with_role(SceneRole.VOID)

    @property
    def divider_object_ids(self) -> tuple[int, ...]:
        return self.objects_with_role(SceneRole.DIVIDER)

    @property
    def frame_object_ids(self) -> tuple[int, ...]:
        return self.objects_with_role(SceneRole.FRAME)

    @property
    def uncertain_object_ids(self) -> tuple[int, ...]:
        return self.objects_with_role(SceneRole.UNCERTAIN)


# =============================================================================
# SMALL HELPERS
# =============================================================================

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _grid_shape(grid: Sequence[Sequence[int]]) -> tuple[int, int]:
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    return height, width


def _object_area(obj: VisualObject) -> int:
    return obj.bbox.height * obj.bbox.width


def _object_grid_fraction(
    obj: VisualObject,
    height: int,
    width: int,
) -> float:
    total = max(1, height * width)
    return obj.cell_count / total


def _bbox_grid_fraction(
    obj: VisualObject,
    height: int,
    width: int,
) -> float:
    total = max(1, height * width)
    return _object_area(obj) / total


def _touches_all_four_borders(
    obj: VisualObject,
    height: int,
    width: int,
) -> bool:
    bbox = obj.bbox
    return (
        bbox.top == 0
        and bbox.left == 0
        and bbox.bottom == height - 1
        and bbox.right == width - 1
    )


def _touches_opposite_borders(
    obj: VisualObject,
    height: int,
    width: int,
) -> bool:
    bbox = obj.bbox

    vertical = bbox.top == 0 and bbox.bottom == height - 1
    horizontal = bbox.left == 0 and bbox.right == width - 1

    return vertical or horizontal


def _is_grid_spanning_line(
    obj: VisualObject,
    height: int,
    width: int,
) -> bool:
    bbox = obj.bbox

    full_vertical = (
        obj.is_vertical_line
        and bbox.top == 0
        and bbox.bottom == height - 1
    )

    full_horizontal = (
        obj.is_horizontal_line
        and bbox.left == 0
        and bbox.right == width - 1
    )

    return full_vertical or full_horizontal


def _is_frame_like(
    obj: VisualObject,
    height: int,
    width: int,
) -> bool:
    bbox = obj.bbox

    if bbox.height < 3 or bbox.width < 3:
        return False

    if not _touches_all_four_borders(obj, height, width):
        return False

    perimeter = 2 * bbox.height + 2 * bbox.width - 4
    return obj.cell_count == perimeter


def _color_cell_counts(
    grid: Sequence[Sequence[int]],
) -> dict[int, int]:
    counts: dict[int, int] = {}

    for row in grid:
        for color in row:
            counts[color] = counts.get(color, 0) + 1

    return counts


def _largest_object_by_color(
    objects: Iterable[VisualObject],
) -> dict[int, VisualObject]:
    largest: dict[int, VisualObject] = {}

    for obj in objects:
        current = largest.get(obj.color)

        if current is None or obj.cell_count > current.cell_count:
            largest[obj.color] = obj

    return largest


# =============================================================================
# ROLE SCORING
# =============================================================================

def _background_score(
    obj: VisualObject,
    *,
    height: int,
    width: int,
    color_fraction: float,
    largest_color_object: bool,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    object_fraction = _object_grid_fraction(obj, height, width)
    bbox_fraction = _bbox_grid_fraction(obj, height, width)

    if object_fraction >= 0.50:
        score += 0.45
        reasons.append("occupies at least half of the grid")
    elif object_fraction >= 0.30:
        score += 0.28
        reasons.append("occupies a large part of the grid")
    elif object_fraction >= 0.15:
        score += 0.12
        reasons.append("occupies a noticeable part of the grid")

    if color_fraction >= 0.50:
        score += 0.30
        reasons.append("its color is the grid majority")
    elif color_fraction >= 0.30:
        score += 0.18
        reasons.append("its color is common across the grid")

    if obj.touches_border:
        score += 0.10
        reasons.append("touches the border")

    if _touches_all_four_borders(obj, height, width):
        score += 0.20
        reasons.append("spans the full grid bounding box")
    elif _touches_opposite_borders(obj, height, width):
        score += 0.08
        reasons.append("touches opposite borders")

    if bbox_fraction >= 0.80:
        score += 0.10
        reasons.append("its bounding box covers most of the grid")

    if largest_color_object:
        score += 0.05
        reasons.append("is the largest component of its color")

    if obj.is_horizontal_line or obj.is_vertical_line:
        score -= 0.30
        reasons.append("line geometry weakens background interpretation")

    if _is_frame_like(obj, height, width):
        score -= 0.45
        reasons.append("frame geometry weakens background interpretation")

    return _clamp(score), reasons


def _divider_score(
    obj: VisualObject,
    *,
    height: int,
    width: int,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if _is_grid_spanning_line(obj, height, width):
        score += 0.90
        reasons.append("is a full-width or full-height line")
    elif obj.is_horizontal_line or obj.is_vertical_line:
        score += 0.45
        reasons.append("is line-shaped")

        if _touches_opposite_borders(obj, height, width):
            score += 0.25
            reasons.append("touches opposite borders")

    if obj.cell_count <= max(height, width):
        score += 0.05
        reasons.append("has line-scale cell count")

    return _clamp(score), reasons


def _frame_score(
    obj: VisualObject,
    *,
    height: int,
    width: int,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if _is_frame_like(obj, height, width):
        score = 0.98
        reasons.append("forms the outer perimeter of the grid")
        return score, reasons

    bbox = obj.bbox

    if (
        bbox.height >= 3
        and bbox.width >= 3
        and obj.touches_border
        and obj.fill_fraction < 0.50
        and _bbox_grid_fraction(obj, height, width) >= 0.50
    ):
        score += 0.48
        reasons.append("is sparse, border-touching, and frame-sized")

    return _clamp(score), reasons


def _void_score(
    obj: VisualObject,
    *,
    height: int,
    width: int,
    background_color: int | None,
    preferred_void_colors: frozenset[int],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if obj.color in preferred_void_colors:
        score += 0.50
        reasons.append("uses a preferred void color")

    if background_color is not None and obj.color == background_color:
        score -= 0.35
        reasons.append("matches the selected background color")

    if not obj.touches_border:
        score += 0.20
        reasons.append("is enclosed away from the border")

    object_fraction = _object_grid_fraction(obj, height, width)

    if object_fraction <= 0.10:
        score += 0.12
        reasons.append("occupies a small portion of the grid")

    if obj.is_solid_rectangle:
        score -= 0.08
        reasons.append("solid rectangle geometry weakens void interpretation")

    return _clamp(score), reasons


def _decoration_score(
    obj: VisualObject,
    *,
    height: int,
    width: int,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    object_fraction = _object_grid_fraction(obj, height, width)

    if obj.cell_count == 1:
        score += 0.55
        reasons.append("is a single cell")
    elif obj.cell_count <= 3:
        score += 0.28
        reasons.append("is extremely small")

    if object_fraction <= 0.03:
        score += 0.15
        reasons.append("occupies very little of the grid")

    return _clamp(score), reasons


# =============================================================================
# MAIN CLASSIFIER
# =============================================================================

def classify_scene(
    grid: Sequence[Sequence[int]],
    detection: ObjectDetectionResult | None = None,
    *,
    connectivity: int = 4,
    preferred_void_colors: Iterable[int] = (0,),
    background_hint: int | None = None,
) -> SceneClassificationResult:
    """
    Classify detected grid components into broad visual roles.

    This module intentionally performs factual visual classification only.
    It does not learn task rules and does not compare input/output pairs.

    Parameters
    ----------
    grid:
        ARC grid.
    detection:
        Existing result from detect_objects(). If omitted, objects are detected.
    connectivity:
        Used only when detection is omitted.
    preferred_void_colors:
        Colors that may represent empty space or holes. ARC black (0) is the
        default, but this remains only a clue, never an absolute rule.
    background_hint:
        Optional known background color from an earlier classifier.

    Returns
    -------
    SceneClassificationResult
    """
    height, width = _grid_shape(grid)

    if height == 0 or width == 0:
        return SceneClassificationResult(
            height=height,
            width=width,
            connectivity=connectivity,
            elements=(),
        )

    if detection is None:
        detection = detect_objects(
            grid,
            connectivity=connectivity,
        )

    objects = tuple(detection.objects)
    color_counts = _color_cell_counts(grid)
    total_cells = max(1, height * width)
    largest_by_color = _largest_object_by_color(objects)
    preferred_void_set = frozenset(preferred_void_colors)

    # First pass: identify the strongest background candidate.
    background_candidates: list[
        tuple[float, VisualObject, tuple[str, ...]]
    ] = []

    for obj in objects:
        color_fraction = color_counts.get(obj.color, 0) / total_cells
        score, reasons = _background_score(
            obj,
            height=height,
            width=width,
            color_fraction=color_fraction,
            largest_color_object=largest_by_color.get(obj.color) is obj,
        )
        background_candidates.append(
            (score, obj, tuple(reasons))
        )

    background_candidates.sort(
        key=lambda item: (
            item[0],
            item[1].cell_count,
            -item[1].object_id,
        ),
        reverse=True,
    )

    selected_background_id: int | None = None
    selected_background_color: int | None = background_hint

    if background_hint is not None:
        hinted = [
            item
            for item in background_candidates
            if item[1].color == background_hint
        ]

        if hinted:
            best_hint = hinted[0]
            selected_background_id = best_hint[1].object_id
            selected_background_color = best_hint[1].color

    if selected_background_id is None and background_candidates:
        best_score, best_obj, _ = background_candidates[0]

        if best_score >= 0.45:
            selected_background_id = best_obj.object_id
            selected_background_color = best_obj.color

    elements: list[SceneElement] = []

    for obj in objects:
        background_score, background_reasons = _background_score(
            obj,
            height=height,
            width=width,
            color_fraction=color_counts.get(obj.color, 0) / total_cells,
            largest_color_object=largest_by_color.get(obj.color) is obj,
        )
        divider_score, divider_reasons = _divider_score(
            obj,
            height=height,
            width=width,
        )
        frame_score, frame_reasons = _frame_score(
            obj,
            height=height,
            width=width,
        )
        void_score, void_reasons = _void_score(
            obj,
            height=height,
            width=width,
            background_color=selected_background_color,
            preferred_void_colors=preferred_void_set,
        )
        decoration_score, decoration_reasons = _decoration_score(
            obj,
            height=height,
            width=width,
        )

        if obj.object_id == selected_background_id:
            role = SceneRole.BACKGROUND
            confidence = max(0.55, background_score)
            reasons = background_reasons

        elif frame_score >= 0.75:
            role = SceneRole.FRAME
            confidence = frame_score
            reasons = frame_reasons

        elif divider_score >= 0.70:
            role = SceneRole.DIVIDER
            confidence = divider_score
            reasons = divider_reasons

        elif (
            void_score >= 0.65
            and obj.color in preferred_void_set
            and obj.color != selected_background_color
        ):
            role = SceneRole.VOID
            confidence = void_score
            reasons = void_reasons

        elif decoration_score >= 0.65:
            role = SceneRole.DECORATION
            confidence = decoration_score
            reasons = decoration_reasons

        else:
            foreground_score = 0.58
            foreground_reasons = ["does not match a stronger structural role"]

            if obj.object_id != selected_background_id:
                foreground_score += 0.10
                foreground_reasons.append("is distinct from the selected background")

            if not obj.is_horizontal_line and not obj.is_vertical_line:
                foreground_score += 0.05
                foreground_reasons.append("has object-like non-line geometry")

            if obj.cell_count > 1:
                foreground_score += 0.04
                foreground_reasons.append("contains multiple cells")

            role = SceneRole.FOREGROUND
            confidence = _clamp(foreground_score)
            reasons = foreground_reasons

        elements.append(
            SceneElement(
                object_id=obj.object_id,
                role=role,
                confidence=round(confidence, 3),
                reasons=tuple(reasons),
            )
        )

    elements.sort(key=lambda element: element.object_id)

    return SceneClassificationResult(
        height=height,
        width=width,
        connectivity=connectivity,
        elements=tuple(elements),
    )


# =============================================================================
# FORMATTING
# =============================================================================

def summarize_scene_roles(
    result: SceneClassificationResult,
) -> dict[str, tuple[int, ...]]:
    summary: dict[str, list[int]] = {}

    for element in result.elements:
        summary.setdefault(element.role.value, []).append(
            element.object_id
        )

    return {
        role: tuple(object_ids)
        for role, object_ids in summary.items()
    }


def format_scene_classification(
    result: SceneClassificationResult,
) -> str:
    if not result.elements:
        return "No scene elements."

    lines = [
        f"Scene: {result.height}x{result.width}",
        f"Connectivity: {result.connectivity}",
        "",
    ]

    for element in result.elements:
        lines.append(
            f"Object {element.object_id}: "
            f"{element.role.value} "
            f"(confidence={element.confidence:.3f})"
        )

        for reason in element.reasons:
            lines.append(f"  - {reason}")

    return "\n".join(lines)


# =============================================================================
# SELF TEST
# =============================================================================

def _self_test() -> None:
    grid = [
        [7, 7, 7, 7, 7, 7],
        [7, 7, 7, 7, 7, 7],
        [7, 7, 7, 7, 7, 7],
        [7, 7, 7, 7, 3, 7],
        [7, 7, 7, 3, 3, 3],
        [7, 7, 7, 7, 3, 0],
    ]

    detection = detect_objects(
        grid,
        connectivity=4,
    )

    result = classify_scene(
        grid,
        detection,
        connectivity=4,
    )

    print("=" * 72)
    print("SCENE CLASSIFIER SELF TEST")
    print("=" * 72)
    print(format_scene_classification(result))
    print()
    print("Role summary:")
    print(summarize_scene_roles(result))


if __name__ == "__main__":
    _self_test()