from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from vision.causal_analyzer import (
    CausalAnalysisReport,
    analyze_rule_causality,
)
from vision.rule_learner import (
    CandidateRule,
    LearnedRule,
    learn_rule,
)
from vision.rule_validator import (
    RuleValidationReport,
    validate_learned_rule,
)
from vision.story_generalizer import (
    GeneralizedEvent,
    GeneralizedPairStory,
    generalize_grid_pair,
)


# =============================================================================
# PRIMITIVE OPERATIONS
# =============================================================================

# Scene preparation / discovery
COPY_INPUT_GRID = "COPY_INPUT_GRID"
CREATE_OUTPUT_GRID = "CREATE_OUTPUT_GRID"
FIND_OBJECT = "FIND_OBJECT"
FIND_OBJECTS = "FIND_OBJECTS"
FIND_COLLECTIONS = "FIND_COLLECTIONS"
FIND_REGIONS = "FIND_REGIONS"
FIND_ANCHORS = "FIND_ANCHORS"
FIND_DIVIDERS = "FIND_DIVIDERS"
FIND_MOTIFS = "FIND_MOTIFS"
BIND_SCENE_ROLES = "BIND_SCENE_ROLES"

# Object preservation and transformation
PRESERVE_OBJECT = "PRESERVE_OBJECT"
MOVE_OBJECT = "MOVE_OBJECT"
ROTATE_OBJECT = "ROTATE_OBJECT"
REFLECT_OBJECT = "REFLECT_OBJECT"
RECOLOR_OBJECT = "RECOLOR_OBJECT"
RESIZE_OBJECT = "RESIZE_OBJECT"
CHANGE_SHAPE = "CHANGE_SHAPE"
CHANGE_ROLE = "CHANGE_ROLE"
CHANGE_INTERNAL_PATTERN = "CHANGE_INTERNAL_PATTERN"
CREATE_OBJECT = "CREATE_OBJECT"
DELETE_OBJECT = "DELETE_OBJECT"
DUPLICATE_OBJECT = "DUPLICATE_OBJECT"
MERGE_OBJECTS = "MERGE_OBJECTS"
SPLIT_OBJECT = "SPLIT_OBJECT"
EXTRACT_OBJECT = "EXTRACT_OBJECT"

# Collection / repetition / lattice operations
CREATE_COLLECTION = "CREATE_COLLECTION"
DELETE_COLLECTION = "DELETE_COLLECTION"
CREATE_REGULAR_ARRAY = "CREATE_REGULAR_ARRAY"
DELETE_REGULAR_ARRAY = "DELETE_REGULAR_ARRAY"
CREATE_REGULAR_REPETITION = "CREATE_REGULAR_REPETITION"
DELETE_REGULAR_REPETITION = "DELETE_REGULAR_REPETITION"
REPEAT_OBJECT = "REPEAT_OBJECT"
TILE_PATTERN = "TILE_PATTERN"
BUILD_LATTICE = "BUILD_LATTICE"
BUILD_SEQUENCE = "BUILD_SEQUENCE"
SORT_COLLECTION = "SORT_COLLECTION"
FILTER_COLLECTION = "FILTER_COLLECTION"
MAP_COLLECTION = "MAP_COLLECTION"

# Grid / canvas operations
RESIZE_GRID = "RESIZE_GRID"
CROP_GRID = "CROP_GRID"
PAD_GRID = "PAD_GRID"
CHANGE_CANVAS = "CHANGE_CANVAS"
CREATE_BACKGROUND = "CREATE_BACKGROUND"
DELETE_BACKGROUND = "DELETE_BACKGROUND"
FILL_BACKGROUND = "FILL_BACKGROUND"
DRAW_BORDER = "DRAW_BORDER"
DRAW_FRAME = "DRAW_FRAME"
DRAW_DIVIDER = "DRAW_DIVIDER"
REMOVE_DIVIDER = "REMOVE_DIVIDER"
PARTITION_GRID = "PARTITION_GRID"

# Region / topology operations
CREATE_REGION = "CREATE_REGION"
DELETE_REGION = "DELETE_REGION"
FILL_REGION = "FILL_REGION"
CLEAR_REGION = "CLEAR_REGION"
COPY_REGION = "COPY_REGION"
MOVE_REGION = "MOVE_REGION"
RECOLOR_REGION = "RECOLOR_REGION"
RESIZE_REGION = "RESIZE_REGION"
EXTRACT_REGION = "EXTRACT_REGION"
ENCLOSE_REGION = "ENCLOSE_REGION"
OPEN_REGION = "OPEN_REGION"
CONNECT_REGIONS = "CONNECT_REGIONS"

# Pattern / motif / symbolic operations
CREATE_MOTIF = "CREATE_MOTIF"
DELETE_MOTIF = "DELETE_MOTIF"
COPY_MOTIF = "COPY_MOTIF"
PLACE_MOTIF = "PLACE_MOTIF"
PROJECT_MOTIF = "PROJECT_MOTIF"
EXPAND_PATTERN = "EXPAND_PATTERN"
COMPRESS_PATTERN = "COMPRESS_PATTERN"
COMPLETE_PATTERN = "COMPLETE_PATTERN"
REPAIR_PATTERN = "REPAIR_PATTERN"
CONTINUE_SEQUENCE = "CONTINUE_SEQUENCE"
SUBSTITUTE_SYMBOL = "SUBSTITUTE_SYMBOL"

# Geometric / structural operations
APPLY_SYMMETRY = "APPLY_SYMMETRY"
MIRROR_SCENE = "MIRROR_SCENE"
ROTATE_SCENE = "ROTATE_SCENE"
TRANSLATE_SCENE = "TRANSLATE_SCENE"
SCALE_SCENE = "SCALE_SCENE"
ALIGN_OBJECTS = "ALIGN_OBJECTS"
DISTRIBUTE_OBJECTS = "DISTRIBUTE_OBJECTS"
STACK_OBJECTS = "STACK_OBJECTS"
PACK_OBJECTS = "PACK_OBJECTS"

# Relationship / constraint operations
SATISFY_RELATIONSHIP = "SATISFY_RELATIONSHIP"
REMOVE_RELATIONSHIP = "REMOVE_RELATIONSHIP"
REVERSE_RELATIONSHIP = "REVERSE_RELATIONSHIP"
PRESERVE_RELATIONSHIP = "PRESERVE_RELATIONSHIP"
SATISFY_CONSTRAINT = "SATISFY_CONSTRAINT"

# Composition / finishing
COMPOSE_SCENE = "COMPOSE_SCENE"
OVERLAY_LAYER = "OVERLAY_LAYER"
RESOLVE_COLLISIONS = "RESOLVE_COLLISIONS"
REBUILD_SCENE = "REBUILD_SCENE"
VERIFY_PLAN_RESULT = "VERIFY_PLAN_RESULT"

# Safe fallback
APPLY_GENERIC_EVENT = "APPLY_GENERIC_EVENT"


# =============================================================================
# EVENT REGISTRY
# =============================================================================

@dataclass(frozen=True)
class PrimitiveSpec:
    operation: str
    target_role: str
    purpose: str
    argument_names: tuple[str, ...] = ()
    selection_needed: bool = False
    relationship_goal: bool = False


def _spec(
    operation: str,
    target_role: str,
    purpose: str,
    *argument_names: str,
    selection_needed: bool = False,
    relationship_goal: bool = False,
) -> PrimitiveSpec:
    return PrimitiveSpec(
        operation=operation,
        target_role=target_role,
        purpose=purpose,
        argument_names=tuple(argument_names),
        selection_needed=selection_needed,
        relationship_goal=relationship_goal,
    )


EVENT_SPECS: dict[str, PrimitiveSpec] = {
    # Preservation
    "object_preservation": _spec(
        PRESERVE_OBJECT,
        "stationary_object",
        "Keep all non-transformed objects unchanged.",
        "preserve",
        selection_needed=True,
    ),
    "scene_preservation": _spec(
        REBUILD_SCENE,
        "output_grid",
        "Preserve the complete scene.",
    ),

    # Object transforms
    "object_translation": _spec(
        MOVE_OBJECT,
        "transformed_object",
        "Move the selected object while preserving identity.",
        "row_delta",
        "column_delta",
        "collision_policy",
        "bounds_policy",
        selection_needed=True,
    ),
    "object_rotation": _spec(
        ROTATE_OBJECT,
        "transformed_object",
        "Rotate the selected object.",
        "rotation",
        "pivot",
        "bounds_policy",
        selection_needed=True,
    ),
    "object_reflection": _spec(
        REFLECT_OBJECT,
        "transformed_object",
        "Reflect the selected object.",
        "axis",
        "pivot",
        "bounds_policy",
        selection_needed=True,
    ),
    "object_color_change": _spec(
        RECOLOR_OBJECT,
        "transformed_object",
        "Apply the learned color mapping.",
        "input_color",
        "output_color",
        "color_mapping",
        selection_needed=True,
    ),
    "object_recoloring": _spec(
        RECOLOR_OBJECT,
        "transformed_object",
        "Apply the learned color mapping.",
        "input_color",
        "output_color",
        "color_mapping",
        selection_needed=True,
    ),
    "object_size_change": _spec(
        RESIZE_OBJECT,
        "transformed_object",
        "Change object size while following the learned geometry.",
        "input_cell_count",
        "output_cell_count",
        "scale_factor",
        "target_dimensions",
        "anchor_policy",
        selection_needed=True,
    ),
    "object_resize": _spec(
        RESIZE_OBJECT,
        "transformed_object",
        "Change object size while following the learned geometry.",
        "scale_factor",
        "target_dimensions",
        "anchor_policy",
        selection_needed=True,
    ),
    "object_dimension_change": _spec(
        RESIZE_OBJECT,
        "transformed_object",
        "Change object dimensions.",
        "input_dimensions",
        "output_dimensions",
        "height_delta",
        "width_delta",
        "anchor_policy",
        selection_needed=True,
    ),
    "object_shape_change": _spec(
        CHANGE_SHAPE,
        "transformed_object",
        "Reconstruct the object in the learned target shape.",
        "input_shape",
        "output_shape",
        "shape_template",
        "anchor_policy",
        selection_needed=True,
    ),
    "scene_role_change": _spec(
        CHANGE_ROLE,
        "transformed_object",
        "Change the object's functional scene role.",
        "input_role",
        "output_role",
        selection_needed=True,
    ),
    "internal_pattern_change": _spec(
        CHANGE_INTERNAL_PATTERN,
        "transformed_object",
        "Change the object's internal cell arrangement.",
        "pattern_rule",
        "source_pattern",
        "target_pattern",
        selection_needed=True,
    ),

    # Object lifecycle
    "object_creation": _spec(
        CREATE_OBJECT,
        "output_object",
        "Create one object required by the learned rule.",
        "color",
        "shape",
        "cell_count",
        "dimensions",
        "placement",
        "scene_role",
        "overlap_policy",
    ),
    "object_deletion": _spec(
        DELETE_OBJECT,
        "input_object",
        "Delete the selected object.",
        "selection_rule",
        selection_needed=True,
    ),
    "object_duplication": _spec(
        DUPLICATE_OBJECT,
        "source_object",
        "Duplicate the selected object.",
        "copy_count",
        "offsets",
        "placement_rule",
        selection_needed=True,
    ),
    "object_merging": _spec(
        MERGE_OBJECTS,
        "object_collection",
        "Merge selected objects into one result.",
        "selection_rule",
        "merge_policy",
        "output_color",
        selection_needed=True,
    ),
    "object_splitting": _spec(
        SPLIT_OBJECT,
        "transformed_object",
        "Split one object into learned components.",
        "split_rule",
        "component_policy",
        selection_needed=True,
    ),
    "object_extraction": _spec(
        EXTRACT_OBJECT,
        "source_object",
        "Extract an object or motif from the scene.",
        "selection_rule",
        "output_canvas_policy",
        selection_needed=True,
    ),

    # Collections
    "object_collection_creation": _spec(
        CREATE_COLLECTION,
        "output_object_collection",
        "Create a collection of similar objects.",
        "count",
        "color",
        "shape",
        "cell_count",
        "dimensions",
        "placements",
        "layout_kind",
        "scene_role",
        "overlap_policy",
    ),
    "object_collection_deletion": _spec(
        DELETE_COLLECTION,
        "input_object_collection",
        "Delete a selected collection of objects.",
        "selection_rule",
        "count",
        selection_needed=True,
    ),
    "regular_array_creation": _spec(
        CREATE_REGULAR_ARRAY,
        "output_object_array",
        "Create a complete regular array of repeated objects.",
        "color",
        "shape",
        "cell_count",
        "object_dimensions",
        "row_count",
        "column_count",
        "row_spacing",
        "column_spacing",
        "origin",
        "top_left_positions",
        "complete_lattice",
        "scene_role",
        "overlap_policy",
    ),
    "regular_array_deletion": _spec(
        DELETE_REGULAR_ARRAY,
        "input_object_array",
        "Delete a regular array.",
        "selection_rule",
        "row_count",
        "column_count",
        "row_spacing",
        "column_spacing",
        selection_needed=True,
    ),
    "regular_repetition_creation": _spec(
        CREATE_REGULAR_REPETITION,
        "output_object_repetition",
        "Create a regular but possibly incomplete repetition.",
        "color",
        "shape",
        "cell_count",
        "object_dimensions",
        "count",
        "row_count",
        "column_count",
        "row_spacing",
        "column_spacing",
        "origin",
        "top_left_positions",
        "complete_lattice",
        "scene_role",
        "mask_or_omission_rule",
        "overlap_policy",
    ),
    "regular_repetition_deletion": _spec(
        DELETE_REGULAR_REPETITION,
        "input_object_repetition",
        "Delete a regular repetition.",
        "selection_rule",
        "count",
        "spacing",
        selection_needed=True,
    ),
    "object_repetition": _spec(
        REPEAT_OBJECT,
        "source_object",
        "Repeat an object according to a learned layout.",
        "copy_count",
        "row_spacing",
        "column_spacing",
        "placement_rule",
        selection_needed=True,
    ),
    "pattern_tiling": _spec(
        TILE_PATTERN,
        "output_grid",
        "Tile a pattern across the output grid.",
        "tile",
        "tile_dimensions",
        "row_period",
        "column_period",
        "phase",
        "mask_rule",
    ),
    "lattice_creation": _spec(
        BUILD_LATTICE,
        "output_grid",
        "Construct a lattice from repeated structural units.",
        "unit_template",
        "row_count",
        "column_count",
        "row_spacing",
        "column_spacing",
        "origin",
        "mask_rule",
    ),

    # Grid / canvas
    "grid_shape_change": _spec(
        RESIZE_GRID,
        "output_grid",
        "Create the learned output-grid dimensions.",
        "input_shape",
        "output_shape",
        "height_delta",
        "width_delta",
        "fill_color",
        "alignment",
    ),
    "grid_resize": _spec(
        RESIZE_GRID,
        "output_grid",
        "Create the learned output-grid dimensions.",
        "input_shape",
        "output_shape",
        "height_delta",
        "width_delta",
        "fill_color",
        "alignment",
    ),
    "grid_crop": _spec(
        CROP_GRID,
        "output_grid",
        "Crop the grid to a learned region.",
        "crop_bbox",
        "selection_rule",
    ),
    "grid_padding": _spec(
        PAD_GRID,
        "output_grid",
        "Pad the grid around the scene.",
        "top_padding",
        "bottom_padding",
        "left_padding",
        "right_padding",
        "fill_color",
    ),
    "canvas_change": _spec(
        CHANGE_CANVAS,
        "output_grid",
        "Apply the learned canvas transformation.",
        "output_shape",
        "background_color",
        "alignment",
    ),
    "background_canvas_creation": _spec(
        CREATE_BACKGROUND,
        "output_canvas",
        "Create the output background canvas.",
        "color",
        "dimensions",
        "fill_policy",
    ),
    "background_canvas_deletion": _spec(
        DELETE_BACKGROUND,
        "input_canvas",
        "Remove or replace the previous background canvas.",
        "selection_rule",
    ),
    "background_canvas_change": _spec(
        FILL_BACKGROUND,
        "output_canvas",
        "Change the canvas background.",
        "input_color",
        "output_color",
        "fill_policy",
    ),

    # Borders / frames / dividers
    "border_creation": _spec(
        DRAW_BORDER,
        "output_grid",
        "Draw a border around a learned region or canvas.",
        "color",
        "thickness",
        "bbox",
        "sides",
    ),
    "frame_creation": _spec(
        DRAW_FRAME,
        "output_grid",
        "Draw a frame around a learned object or region.",
        "color",
        "thickness",
        "bbox",
        "margin",
    ),
    "divider_creation": _spec(
        DRAW_DIVIDER,
        "output_grid",
        "Create a divider line or separator.",
        "color",
        "orientation",
        "position",
        "thickness",
        "extent",
    ),
    "divider_deletion": _spec(
        REMOVE_DIVIDER,
        "output_grid",
        "Remove a divider.",
        "selection_rule",
    ),
    "grid_partition": _spec(
        PARTITION_GRID,
        "output_grid",
        "Partition the grid into learned sections.",
        "row_dividers",
        "column_dividers",
        "divider_color",
        "partition_policy",
    ),

    # Regions
    "region_creation": _spec(
        CREATE_REGION,
        "output_region",
        "Create a region.",
        "color",
        "cells",
        "bbox",
        "shape",
        "placement",
    ),
    "region_deletion": _spec(
        DELETE_REGION,
        "input_region",
        "Delete a region.",
        "selection_rule",
        selection_needed=True,
    ),
    "region_fill": _spec(
        FILL_REGION,
        "target_region",
        "Fill a selected region.",
        "selection_rule",
        "fill_color",
        "fill_policy",
        selection_needed=True,
    ),
    "region_clear": _spec(
        CLEAR_REGION,
        "target_region",
        "Clear a selected region.",
        "selection_rule",
        "replacement_color",
        selection_needed=True,
    ),
    "region_copy": _spec(
        COPY_REGION,
        "source_region",
        "Copy a region.",
        "selection_rule",
        "placement",
        selection_needed=True,
    ),
    "region_translation": _spec(
        MOVE_REGION,
        "transformed_region",
        "Move a selected region.",
        "row_delta",
        "column_delta",
        selection_needed=True,
    ),
    "region_recoloring": _spec(
        RECOLOR_REGION,
        "transformed_region",
        "Recolor a selected region.",
        "input_color",
        "output_color",
        selection_needed=True,
    ),
    "region_resize": _spec(
        RESIZE_REGION,
        "transformed_region",
        "Resize a selected region.",
        "target_dimensions",
        "scale_factor",
        selection_needed=True,
    ),
    "region_extraction": _spec(
        EXTRACT_REGION,
        "source_region",
        "Extract a selected region.",
        "selection_rule",
        "output_canvas_policy",
        selection_needed=True,
    ),
    "region_enclosure": _spec(
        ENCLOSE_REGION,
        "target_region",
        "Enclose a selected region.",
        "border_color",
        "thickness",
        "margin",
        selection_needed=True,
    ),
    "region_opening": _spec(
        OPEN_REGION,
        "target_region",
        "Open or puncture a selected enclosure.",
        "opening_rule",
        "replacement_color",
        selection_needed=True,
    ),
    "region_connection": _spec(
        CONNECT_REGIONS,
        "related_regions",
        "Connect selected regions.",
        "connection_color",
        "path_policy",
        "thickness",
        selection_needed=True,
    ),

    # Motifs / patterns
    "motif_creation": _spec(
        CREATE_MOTIF,
        "output_motif",
        "Create a learned motif.",
        "template",
        "color_mapping",
        "placement",
    ),
    "motif_deletion": _spec(
        DELETE_MOTIF,
        "input_motif",
        "Delete a selected motif.",
        "selection_rule",
        selection_needed=True,
    ),
    "motif_copy": _spec(
        COPY_MOTIF,
        "source_motif",
        "Copy a learned motif.",
        "selection_rule",
        "placements",
        selection_needed=True,
    ),
    "motif_placement": _spec(
        PLACE_MOTIF,
        "output_motif",
        "Place a motif at learned locations.",
        "template",
        "placements",
        "orientation_policy",
    ),
    "motif_projection": _spec(
        PROJECT_MOTIF,
        "source_motif",
        "Project a motif into another region or canvas.",
        "selection_rule",
        "projection_rule",
        "target_region",
        selection_needed=True,
    ),
    "pattern_expansion": _spec(
        EXPAND_PATTERN,
        "source_pattern",
        "Expand a pattern.",
        "expansion_rule",
        "output_shape",
        "period",
        "phase",
    ),
    "pattern_compression": _spec(
        COMPRESS_PATTERN,
        "source_pattern",
        "Compress a pattern.",
        "compression_rule",
        "output_shape",
    ),
    "pattern_completion": _spec(
        COMPLETE_PATTERN,
        "incomplete_pattern",
        "Complete a missing pattern.",
        "completion_rule",
        "missing_cells",
        "symmetry_or_period",
    ),
    "pattern_repair": _spec(
        REPAIR_PATTERN,
        "damaged_pattern",
        "Repair a damaged pattern.",
        "repair_rule",
        "damaged_cells",
        "source_template",
    ),
    "sequence_continuation": _spec(
        CONTINUE_SEQUENCE,
        "source_sequence",
        "Continue a learned sequence.",
        "sequence_rule",
        "step_count",
        "placement_rule",
    ),
    "symbol_substitution": _spec(
        SUBSTITUTE_SYMBOL,
        "symbolic_structure",
        "Substitute one symbolic pattern for another.",
        "input_symbol",
        "output_symbol",
        "mapping_rule",
    ),

    # Symmetry / geometry
    "symmetry_completion": _spec(
        APPLY_SYMMETRY,
        "output_grid",
        "Complete the scene using learned symmetry.",
        "symmetry_type",
        "axis",
        "center",
        "source_side",
    ),
    "scene_reflection": _spec(
        MIRROR_SCENE,
        "output_grid",
        "Mirror the complete scene.",
        "axis",
        "pivot",
    ),
    "scene_rotation": _spec(
        ROTATE_SCENE,
        "output_grid",
        "Rotate the complete scene.",
        "rotation",
        "pivot",
    ),
    "scene_translation": _spec(
        TRANSLATE_SCENE,
        "output_grid",
        "Translate the complete scene.",
        "row_delta",
        "column_delta",
    ),
    "scene_scaling": _spec(
        SCALE_SCENE,
        "output_grid",
        "Scale the complete scene.",
        "scale_factor",
        "output_shape",
        "anchor_policy",
    ),
    "object_alignment": _spec(
        ALIGN_OBJECTS,
        "object_collection",
        "Align selected objects.",
        "alignment_axis",
        "alignment_edge",
        "spacing_policy",
        selection_needed=True,
    ),
    "object_distribution": _spec(
        DISTRIBUTE_OBJECTS,
        "object_collection",
        "Distribute selected objects evenly.",
        "axis",
        "spacing",
        "bounds",
        selection_needed=True,
    ),
    "object_stacking": _spec(
        STACK_OBJECTS,
        "object_collection",
        "Stack selected objects.",
        "direction",
        "spacing",
        "order_rule",
        selection_needed=True,
    ),
    "object_packing": _spec(
        PACK_OBJECTS,
        "object_collection",
        "Pack selected objects into a target region.",
        "target_region",
        "packing_rule",
        "spacing",
        selection_needed=True,
    ),

    # Relationships
    "relationship_creation": _spec(
        SATISFY_RELATIONSHIP,
        "related_objects",
        "Adjust objects until the required relationship exists.",
        "relationship",
        "subject_selection",
        "object_selection",
        "placement_or_adjustment",
        selection_needed=True,
        relationship_goal=True,
    ),
    "relationship_deletion": _spec(
        REMOVE_RELATIONSHIP,
        "related_objects",
        "Adjust objects until the old relationship no longer exists.",
        "relationship",
        "subject_selection",
        "object_selection",
        "placement_or_adjustment",
        selection_needed=True,
        relationship_goal=True,
    ),
    "relationship_reversal": _spec(
        REVERSE_RELATIONSHIP,
        "related_objects",
        "Reverse a directional relationship.",
        "old_relationship",
        "new_relationship",
        "subject_selection",
        "object_selection",
        "placement_or_adjustment",
        selection_needed=True,
        relationship_goal=True,
    ),
    "relationship_change": _spec(
        SATISFY_RELATIONSHIP,
        "related_objects",
        "Apply a learned relationship constraint.",
        "input_relationship",
        "output_relationship",
        "placement_or_adjustment",
        selection_needed=True,
        relationship_goal=True,
    ),
    "relationship_preservation": _spec(
        PRESERVE_RELATIONSHIP,
        "related_objects",
        "Preserve a learned relationship.",
        "relationship",
        "subject_selection",
        "object_selection",
        selection_needed=True,
        relationship_goal=True,
    ),
}


# =============================================================================
# PLAN DATA TYPES
# =============================================================================

@dataclass(frozen=True)
class PlanArgument:
    name: str
    value: Any
    source: str
    resolved: bool


@dataclass(frozen=True)
class PlanStep:
    step_number: int
    operation: str
    target_role: str
    arguments: tuple[PlanArgument, ...]
    purpose: str
    executable: bool
    source_event_type: str | None = None


@dataclass(frozen=True)
class RulePlan:
    rule_name: str
    validation_verdict: str
    steps: tuple[PlanStep, ...]
    unresolved_values: tuple[str, ...]
    supported_operations: tuple[str, ...]
    unsupported_events: tuple[str, ...]
    ready_for_execution: bool
    warnings: tuple[str, ...]


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                ((str(key), _freeze(item)) for key, item in value.items()),
                key=lambda item: item[0],
            )
        )

    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)

    if isinstance(value, set):
        return tuple(sorted((_freeze(item) for item in value), key=repr))

    try:
        hash(value)
    except TypeError:
        return repr(value)

    return value


def _event_properties(event: GeneralizedEvent) -> dict[str, Any]:
    raw = getattr(event, "properties", ())

    if isinstance(raw, Mapping):
        return dict(raw)

    try:
        return dict(raw)
    except (TypeError, ValueError):
        return {}


def _arg(
    name: str,
    value: Any,
    *,
    source: str,
    resolved: bool,
) -> PlanArgument:
    return PlanArgument(
        name=name,
        value=_freeze(value),
        source=source,
        resolved=resolved,
    )


def _append_step(
    steps: list[PlanStep],
    operation: str,
    target_role: str,
    arguments: Iterable[PlanArgument],
    purpose: str,
    *,
    source_event_type: str | None = None,
) -> None:
    argument_tuple = tuple(arguments)

    steps.append(
        PlanStep(
            step_number=len(steps) + 1,
            operation=operation,
            target_role=target_role,
            arguments=argument_tuple,
            purpose=purpose,
            executable=all(argument.resolved for argument in argument_tuple),
            source_event_type=source_event_type,
        )
    )


def _events_by_type(
    stories: Sequence[GeneralizedPairStory],
) -> dict[str, list[GeneralizedEvent]]:
    grouped: dict[str, list[GeneralizedEvent]] = {}

    for story in stories:
        for event in story.events:
            grouped.setdefault(event.event_type, []).append(event)

    return grouped


def _events_per_pair(
    stories: Sequence[GeneralizedPairStory],
    event_type: str,
) -> tuple[tuple[GeneralizedEvent, ...], ...]:
    return tuple(
        tuple(
            event
            for event in story.events
            if event.event_type == event_type
        )
        for story in stories
    )


def _shared_event_type_counts(
    stories: Sequence[GeneralizedPairStory],
) -> dict[str, int]:
    if not stories:
        return {}

    counters = [
        Counter(story.event_types)
        for story in stories
    ]

    shared = counters[0].copy()

    for counter in counters[1:]:
        for event_type in list(shared):
            shared[event_type] = min(
                shared[event_type],
                counter.get(event_type, 0),
            )

            if shared[event_type] <= 0:
                del shared[event_type]

    return dict(shared)


def _common_exact_value(
    stories: Sequence[GeneralizedPairStory],
    event_type: str,
    property_name: str,
) -> tuple[bool, Any, str]:
    """
    Resolve only when every train pair supplies the same concrete value.

    Multiple events of the same type within one pair are represented as a
    tuple in deterministic order. This retains multiplicity.
    """
    per_pair_values: list[Any] = []

    for pair_events in _events_per_pair(stories, event_type):
        values = [
            _freeze(_event_properties(event).get(property_name))
            for event in pair_events
            if property_name in _event_properties(event)
        ]

        if not values:
            return (
                False,
                None,
                f"{property_name} is not present in every train pair",
            )

        if len(values) == 1:
            per_pair_values.append(values[0])
        else:
            per_pair_values.append(tuple(sorted(values, key=repr)))

    if per_pair_values and all(
        value == per_pair_values[0]
        for value in per_pair_values[1:]
    ):
        return (
            True,
            per_pair_values[0],
            "same concrete value in every train pair",
        )

    return (
        False,
        None,
        "value varies across train pairs or requires scene inference",
    )


def _observed_values(
    stories: Sequence[GeneralizedPairStory],
    event_type: str,
    property_name: str,
) -> tuple[Any, ...]:
    values: list[Any] = []

    for story in stories:
        for event in story.events:
            if event.event_type != event_type:
                continue

            properties = _event_properties(event)

            if property_name in properties:
                values.append(_freeze(properties[property_name]))

    return tuple(values)


def _argument_for_property(
    stories: Sequence[GeneralizedPairStory],
    event_type: str,
    property_name: str,
) -> PlanArgument:
    resolved, value, reason = _common_exact_value(
        stories,
        event_type,
        property_name,
    )

    if resolved:
        return _arg(
            property_name,
            value,
            source=f"training evidence: {reason}",
            resolved=True,
        )

    observations = _observed_values(
        stories,
        event_type,
        property_name,
    )

    source = (
        "infer from training variation and the test scene"
        if observations
        else "infer from the learned event and test scene"
    )

    if observations:
        source += f"; observed={observations!r}"

    return _arg(
        property_name,
        None,
        source=source,
        resolved=False,
    )


def _selection_argument(event_type: str) -> PlanArgument:
    return _arg(
        "selection_rule",
        None,
        source=(
            f"learn the source/target role binding for {event_type} "
            "from training objects and the test scene"
        ),
        resolved=False,
    )


def _arguments_for_event(
    stories: Sequence[GeneralizedPairStory],
    event_type: str,
    spec: PrimitiveSpec,
    candidate: CandidateRule,
) -> tuple[PlanArgument, ...]:
    arguments: list[PlanArgument] = []

    for property_name in spec.argument_names:
        if property_name == "preserve":
            arguments.append(
                _arg(
                    "preserve",
                    ("position", "shape", "color", "size", "identity"),
                    source="learned preservation facts",
                    resolved=True,
                )
            )
            continue

        if property_name == "selection_rule":
            arguments.append(_selection_argument(event_type))
            continue

        if property_name in {
            "collision_policy",
            "bounds_policy",
            "overlap_policy",
        }:
            arguments.append(
                _arg(
                    property_name,
                    "reject_invalid",
                    source="safe executor default",
                    resolved=True,
                )
            )
            continue

        arguments.append(
            _argument_for_property(
                stories,
                event_type,
                property_name,
            )
        )

    if spec.selection_needed and not any(
        argument.name == "selection_rule"
        for argument in arguments
    ):
        arguments.insert(0, _selection_argument(event_type))

    # Preserve explicit learner metadata for downstream inference.
    if event_type == "object_translation":
        variable_facts = set(candidate.variable_facts)

        arguments.append(
            _arg(
                "distance_policy",
                (
                    "infer_from_test_scene"
                    if "moving_object.translation_distance varies"
                    in variable_facts
                    else "fixed_or_learned"
                ),
                source="learned rule variability",
                resolved=True,
            )
        )

    return tuple(arguments)


def _slot_exact_properties(slot: Any) -> dict[str, Any]:
    try:
        return dict(slot.exact_properties)
    except (AttributeError, TypeError, ValueError):
        return {}


def _slot_variable_properties(slot: Any) -> set[str]:
    try:
        return set(slot.variable_properties)
    except (AttributeError, TypeError):
        return set()


def _arguments_for_slot(
    stories: Sequence[GeneralizedPairStory],
    slot: Any,
    spec: PrimitiveSpec,
    candidate: CandidateRule,
) -> tuple[PlanArgument, ...]:
    """
    Build one operation for one learned structural role.

    Fixed slot properties are passed through as resolved arguments.
    Variable slot properties remain unresolved for value inference.
    """
    event_type = str(slot.event_type)
    slot_name = str(slot.slot_name)
    exact = _slot_exact_properties(slot)
    variable = _slot_variable_properties(slot)

    arguments: list[PlanArgument] = [
        _arg(
            "learned_role",
            slot_name,
            source="rule learner structural role",
            resolved=True,
        ),
        _arg(
            "role_signature",
            getattr(slot, "structural_signature", ()),
            source="rule learner role correspondence",
            resolved=True,
        ),
    ]

    for property_name in spec.argument_names:
        if property_name == "preserve":
            arguments.append(
                _arg(
                    "preserve",
                    ("position", "shape", "color", "size", "identity"),
                    source="learned preservation facts",
                    resolved=True,
                )
            )
            continue

        if property_name == "selection_rule":
            arguments.append(
                _arg(
                    "selection_rule",
                    {
                        "learned_role": slot_name,
                        "fixed_properties": exact,
                    },
                    source="learned structural role and fixed properties",
                    resolved=True,
                )
            )
            continue

        if property_name in {
            "collision_policy",
            "bounds_policy",
            "overlap_policy",
        }:
            arguments.append(
                _arg(
                    property_name,
                    "reject_invalid",
                    source="safe executor default",
                    resolved=True,
                )
            )
            continue

        if property_name in exact:
            arguments.append(
                _arg(
                    property_name,
                    exact[property_name],
                    source=f"fixed property of learned role {slot_name}",
                    resolved=True,
                )
            )
            continue

        if property_name in variable:
            arguments.append(
                _arg(
                    property_name,
                    None,
                    source=(
                        f"property varies for learned role {slot_name}; "
                        "infer from the test scene"
                    ),
                    resolved=False,
                )
            )
            continue

        arguments.append(
            _argument_for_property(
                stories,
                event_type,
                property_name,
            )
        )

    if spec.selection_needed and not any(
        argument.name == "selection_rule"
        for argument in arguments
    ):
        arguments.insert(
            2,
            _arg(
                "selection_rule",
                {
                    "learned_role": slot_name,
                    "fixed_properties": exact,
                },
                source="learned structural role and fixed properties",
                resolved=True,
            ),
        )

    if event_type == "object_translation":
        variable_facts = set(candidate.variable_facts)

        arguments.append(
            _arg(
                "distance_policy",
                (
                    "infer_from_test_scene"
                    if "moving_object.translation_distance varies"
                    in variable_facts
                    else "fixed_or_learned"
                ),
                source="learned rule variability",
                resolved=True,
            )
        )

    return tuple(arguments)


# =============================================================================
# FALLBACK PLANNING FOR FUTURE EVENTS
# =============================================================================

def _fallback_spec(event_type: str) -> PrimitiveSpec:
    lowered = event_type.lower()

    if "array" in lowered:
        return _spec(
            CREATE_REGULAR_ARRAY
            if any(word in lowered for word in ("creation", "create", "added"))
            else DELETE_REGULAR_ARRAY,
            "array_structure",
            f"Apply future array event: {event_type}.",
            "template",
            "row_count",
            "column_count",
            "row_spacing",
            "column_spacing",
            "placement",
        )

    if any(word in lowered for word in ("repetition", "repeat", "lattice")):
        return _spec(
            CREATE_REGULAR_REPETITION,
            "repeated_structure",
            f"Apply future repetition event: {event_type}.",
            "template",
            "count",
            "spacing",
            "placement_rule",
        )

    if "collection" in lowered:
        return _spec(
            CREATE_COLLECTION
            if any(word in lowered for word in ("creation", "create", "added"))
            else DELETE_COLLECTION,
            "object_collection",
            f"Apply future collection event: {event_type}.",
            "selection_rule",
            "count",
            "layout_rule",
        )

    if "region" in lowered:
        return _spec(
            APPLY_GENERIC_EVENT,
            "region_structure",
            f"Apply future region event: {event_type}.",
            "event_properties",
            "selection_rule",
        )

    if any(word in lowered for word in ("pattern", "motif", "symbol")):
        return _spec(
            APPLY_GENERIC_EVENT,
            "pattern_structure",
            f"Apply future pattern event: {event_type}.",
            "event_properties",
            "template",
            "placement_rule",
        )

    if any(word in lowered for word in ("grid", "canvas", "background")):
        return _spec(
            APPLY_GENERIC_EVENT,
            "output_grid",
            f"Apply future grid/canvas event: {event_type}.",
            "event_properties",
        )

    return _spec(
        APPLY_GENERIC_EVENT,
        "scene_structure",
        f"Preserve and execute future event without dropping it: {event_type}.",
        "event_properties",
    )


def _fallback_arguments(
    stories: Sequence[GeneralizedPairStory],
    event_type: str,
) -> tuple[PlanArgument, ...]:
    pair_property_sets: list[tuple[tuple[str, Any], ...]] = []

    for story in stories:
        matching = [
            event
            for event in story.events
            if event.event_type == event_type
        ]

        if not matching:
            continue

        pair_property_sets.append(
            tuple(
                sorted(
                    (
                        (
                            name,
                            _freeze(value),
                        )
                        for event in matching
                        for name, value in _event_properties(event).items()
                    ),
                    key=lambda item: (item[0], repr(item[1])),
                )
            )
        )

    resolved = (
        bool(pair_property_sets)
        and all(
            value == pair_property_sets[0]
            for value in pair_property_sets[1:]
        )
    )

    return (
        _arg(
            "event_properties",
            pair_property_sets[0] if resolved else None,
            source=(
                "same complete property set in every train pair"
                if resolved
                else f"preserved future-event evidence: {pair_property_sets!r}"
            ),
            resolved=resolved,
        ),
    )


# =============================================================================
# PLAN BUILDING
# =============================================================================

DERIVED_RELATIONSHIP_EVENTS = {
    "relationship_creation",
    "relationship_deletion",
    "relationship_reversal",
    "relationship_change",
}

DISCOVERY_OPERATIONS = {
    FIND_OBJECT,
    FIND_OBJECTS,
    FIND_COLLECTIONS,
    FIND_REGIONS,
    FIND_ANCHORS,
    FIND_DIVIDERS,
    FIND_MOTIFS,
    BIND_SCENE_ROLES,
}


def _ordered_required_event_types(
    candidate: CandidateRule,
    stories: Sequence[GeneralizedPairStory],
    causal_report: CausalAnalysisReport,
) -> tuple[str, ...]:
    required = list(candidate.required_event_types)

    # Preserve shared structural events even when the current learner does not
    # yet know their vocabulary.
    shared_counts = _shared_event_type_counts(stories)

    for event_type in shared_counts:
        if event_type in required:
            continue

        if event_type in DERIVED_RELATIONSHIP_EVENTS:
            continue

        required.append(event_type)

    return tuple(dict.fromkeys(required))


def _needed_discovery_steps(
    event_types: Sequence[str],
) -> tuple[str, ...]:
    needed: list[str] = [FIND_OBJECTS, BIND_SCENE_ROLES]

    if any(
        event_type.startswith("regular_")
        or "collection" in event_type
        or "repetition" in event_type
        or "array" in event_type
        for event_type in event_types
    ):
        needed.append(FIND_COLLECTIONS)

    if any("region" in event_type for event_type in event_types):
        needed.append(FIND_REGIONS)

    if any(
        word in event_type
        for event_type in event_types
        for word in ("motif", "pattern", "symbol")
    ):
        needed.append(FIND_MOTIFS)

    if any("divider" in event_type for event_type in event_types):
        needed.append(FIND_DIVIDERS)

    if any(
        event_type in DERIVED_RELATIONSHIP_EVENTS
        for event_type in event_types
    ):
        needed.append(FIND_ANCHORS)

    return tuple(dict.fromkeys(needed))


def build_rule_plan(
    learned: LearnedRule,
    validation: RuleValidationReport,
    stories: Sequence[GeneralizedPairStory],
) -> RulePlan:
    warnings: list[str] = []
    unsupported_events: list[str] = []
    steps: list[PlanStep] = []

    if learned.best_rule is None:
        return RulePlan(
            rule_name="No learned rule",
            validation_verdict=validation.verdict,
            steps=(),
            unresolved_values=(),
            supported_operations=(),
            unsupported_events=(),
            ready_for_execution=False,
            warnings=(
                learned.warning
                or "No best rule was available for planning.",
            ),
        )

    candidate = learned.best_rule

    if validation.verdict != "PASS":
        warnings.append(
            "The learned rule did not pass validation. "
            "The plan must not be executed."
        )

    causal_report = analyze_rule_causality(
        candidate,
        stories,
    )

    required_types = _ordered_required_event_types(
        candidate,
        stories,
        causal_report,
    )

    _append_step(
        steps,
        COPY_INPUT_GRID,
        "input_grid",
        (),
        "Start from an unchanged copy of the test input.",
    )

    for operation in _needed_discovery_steps(required_types):
        if operation == FIND_OBJECTS:
            arguments = (
                _arg(
                    "connectivity",
                    4,
                    source="current scene-analysis policy",
                    resolved=True,
                ),
            )
            target = "scene_objects"
            purpose = "Detect concrete objects in the test scene."
        elif operation == BIND_SCENE_ROLES:
            arguments = (
                _arg(
                    "binding_rule",
                    None,
                    source=(
                        "learn transformed, stationary, source, target, "
                        "anchor, canvas, motif, region, and collection roles"
                    ),
                    resolved=False,
                ),
            )
            target = "scene_roles"
            purpose = "Bind learned semantic roles to test-scene structures."
        elif operation == FIND_COLLECTIONS:
            arguments = (
                _arg(
                    "grouping_signature",
                    (
                        "color",
                        "shape",
                        "cell_count",
                        "dimensions",
                        "scene_role",
                    ),
                    source="collection-analysis policy",
                    resolved=True,
                ),
                _arg(
                    "detect_regular_layouts",
                    True,
                    source="collection-analysis policy",
                    resolved=True,
                ),
            )
            target = "scene_collections"
            purpose = "Detect object groups, arrays, repetitions, and lattices."
        elif operation == FIND_REGIONS:
            arguments = ()
            target = "scene_regions"
            purpose = "Detect connected, enclosed, bordered, and void regions."
        elif operation == FIND_MOTIFS:
            arguments = ()
            target = "scene_motifs"
            purpose = "Detect repeated or reusable motifs and symbolic units."
        elif operation == FIND_DIVIDERS:
            arguments = ()
            target = "scene_dividers"
            purpose = "Detect divider lines and partition structure."
        elif operation == FIND_ANCHORS:
            arguments = ()
            target = "scene_anchors"
            purpose = "Detect relationship anchors and placement references."
        else:
            arguments = ()
            target = "scene"
            purpose = f"Perform discovery operation {operation}."

        _append_step(
            steps,
            operation,
            target,
            arguments,
            purpose,
        )

    planned_slot_event_types: set[str] = set()

    for slot in candidate.required_event_slots:
        event_type = slot.event_type
        planned_slot_event_types.add(event_type)

        spec = EVENT_SPECS.get(event_type)

        if spec is None:
            spec = _fallback_spec(event_type)
            warnings.append(
                f"Role {slot.slot_name!r} uses the generic future-event planner."
            )

        if spec.operation == APPLY_GENERIC_EVENT:
            arguments = (
                _arg(
                    "learned_role",
                    slot.slot_name,
                    source="rule learner structural role",
                    resolved=True,
                ),
                _arg(
                    "event_properties",
                    dict(slot.exact_properties),
                    source="fixed properties of learned role",
                    resolved=True,
                ),
                _arg(
                    "variable_properties",
                    tuple(slot.variable_properties),
                    source="variable properties of learned role",
                    resolved=True,
                ),
            )
        else:
            arguments = _arguments_for_slot(
                stories,
                slot,
                spec,
                candidate,
            )

        _append_step(
            steps,
            spec.operation,
            slot.slot_name,
            arguments,
            f"{spec.purpose} Learned role: {slot.slot_name}.",
            source_event_type=event_type,
        )

    # Some event types may not yet have a learned role. Preserve those through
    # the existing event-level planner instead of dropping them.
    for event_type in required_types:
        if event_type in planned_slot_event_types:
            continue

        spec = EVENT_SPECS.get(event_type)

        if spec is None:
            spec = _fallback_spec(event_type)
            warnings.append(
                f"Event {event_type!r} uses the generic future-event planner."
            )

        if spec.operation == APPLY_GENERIC_EVENT:
            arguments = _fallback_arguments(
                stories,
                event_type,
            )
        else:
            arguments = _arguments_for_event(
                stories,
                event_type,
                spec,
                candidate,
            )

        _append_step(
            steps,
            spec.operation,
            spec.target_role,
            arguments,
            spec.purpose,
            source_event_type=event_type,
        )

    goal_types = {
        event.event_type
        for pair_analysis in causal_report.pair_analyses
        for event in pair_analysis.goals_or_constraints
    }

    for event_type in sorted(goal_types):
        if event_type in required_types:
            continue

        spec = EVENT_SPECS.get(
            event_type,
            _fallback_spec(event_type),
        )

        arguments = (
            _fallback_arguments(stories, event_type)
            if spec.operation == APPLY_GENERIC_EVENT
            else _arguments_for_event(
                stories,
                event_type,
                spec,
                candidate,
            )
        )

        _append_step(
            steps,
            spec.operation,
            spec.target_role,
            arguments,
            (
                "Use this event as an output goal or placement constraint, "
                "not as an independently replayed visual side effect."
            ),
            source_event_type=event_type,
        )

    _append_step(
        steps,
        COMPOSE_SCENE,
        "output_grid",
        (
            _arg(
                "layer_order",
                None,
                source=(
                    "infer background, structural, collection, object, motif, "
                    "and repair overlay order from training"
                ),
                resolved=False,
            ),
            _arg(
                "collision_policy",
                "reject_invalid",
                source="safe executor default",
                resolved=True,
            ),
        ),
        "Compose all generated structures into one output grid.",
    )

    _append_step(
        steps,
        REBUILD_SCENE,
        "output_grid",
        (),
        (
            "Recompute objects and relationships from final geometry instead "
            "of replaying derived scene-graph side effects."
        ),
    )

    _append_step(
        steps,
        VERIFY_PLAN_RESULT,
        "output_grid",
        (
            _arg(
                "required_events",
                required_types,
                source="learned rule plus shared structural evidence",
                resolved=True,
            ),
            _arg(
                "preserved_facts",
                tuple(candidate.preserved_facts),
                source="learned rule",
                resolved=True,
            ),
            _arg(
                "goal_constraints",
                tuple(sorted(goal_types)),
                source="causal analysis",
                resolved=True,
            ),
            _arg(
                "verify_shape",
                True,
                source="verification policy",
                resolved=True,
            ),
            _arg(
                "verify_objects",
                True,
                source="verification policy",
                resolved=True,
            ),
            _arg(
                "verify_collections",
                True,
                source="verification policy",
                resolved=True,
            ),
            _arg(
                "verify_relationships",
                True,
                source="verification policy",
                resolved=True,
            ),
        ),
        "Verify that the generated grid satisfies all learned evidence.",
    )

    unresolved_values = tuple(
        dict.fromkeys(
            f"step {step.step_number}: {argument.name}"
            for step in steps
            for argument in step.arguments
            if not argument.resolved
        )
    )

    supported_operations = tuple(
        dict.fromkeys(step.operation for step in steps)
    )

    if unsupported_events:
        warnings.append(
            "Some events have no planning representation: "
            + ", ".join(sorted(set(unsupported_events)))
        )

    if unresolved_values:
        warnings.append(
            "The planner preserved every known operation, but unresolved "
            "selection, placement, composition, or test-specific values still "
            "require value inference and scene resolution."
        )

    ready_for_execution = (
        validation.verdict == "PASS"
        and not unsupported_events
        and not unresolved_values
    )

    return RulePlan(
        rule_name=candidate.name,
        validation_verdict=validation.verdict,
        steps=tuple(steps),
        unresolved_values=unresolved_values,
        supported_operations=supported_operations,
        unsupported_events=tuple(sorted(set(unsupported_events))),
        ready_for_execution=ready_for_execution,
        warnings=tuple(dict.fromkeys(warnings)),
    )


# =============================================================================
# PIPELINE HELPERS
# =============================================================================

def learn_validate_and_plan(
    stories: Sequence[GeneralizedPairStory],
) -> tuple[LearnedRule, RuleValidationReport, RulePlan]:
    learned = learn_rule(stories)
    validation = validate_learned_rule(learned, stories)
    plan = build_rule_plan(learned, validation, stories)

    return learned, validation, plan


def learn_validate_and_plan_grid_pairs(
    train_pairs: Sequence[
        tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]
    ],
    *,
    connectivity: int = 4,
    include_preservations: bool = True,
) -> tuple[LearnedRule, RuleValidationReport, RulePlan]:
    stories = tuple(
        generalize_grid_pair(
            input_grid,
            output_grid,
            connectivity=connectivity,
            include_preservations=include_preservations,
        )
        for input_grid, output_grid in train_pairs
    )

    return learn_validate_and_plan(stories)


# =============================================================================
# FORMATTING
# =============================================================================

def format_plan_step(step: PlanStep) -> str:
    state = "READY" if step.executable else "NEEDS INFERENCE"

    lines = [
        f"STEP {step.step_number}: {step.operation}",
        f"  Target : {step.target_role}",
        f"  State  : {state}",
        f"  Purpose: {step.purpose}",
    ]

    if step.source_event_type:
        lines.append(
            f"  Event  : {step.source_event_type}"
        )

    if step.arguments:
        lines.append("  Arguments:")

        for argument in step.arguments:
            value = (
                repr(argument.value)
                if argument.resolved
                else "<UNRESOLVED>"
            )

            lines.append(
                f"    - {argument.name}: {value}"
            )
            lines.append(
                f"      source: {argument.source}"
            )

    return "\n".join(lines)


def format_rule_plan(plan: RulePlan) -> str:
    lines = [
        "=" * 72,
        "RULE PLAN",
        "=" * 72,
        f"Rule              : {plan.rule_name}",
        f"Validation verdict: {plan.validation_verdict}",
        (
            "Ready to execute  : "
            + ("YES" if plan.ready_for_execution else "NO")
        ),
        "",
    ]

    for step in plan.steps:
        lines.append(format_plan_step(step))
        lines.append("")

    if plan.unresolved_values:
        lines.extend([
            "UNRESOLVED TEST-SPECIFIC VALUES",
            "-" * 72,
        ])

        for value in plan.unresolved_values:
            lines.append(f"  ? {value}")

        lines.append("")

    if plan.unsupported_events:
        lines.extend([
            "UNSUPPORTED EVENTS",
            "-" * 72,
        ])

        for event_type in plan.unsupported_events:
            lines.append(f"  ! {event_type}")

        lines.append("")

    if plan.warnings:
        lines.extend([
            "WARNINGS",
            "-" * 72,
        ])

        for warning in plan.warnings:
            lines.append(f"  ! {warning}")

        lines.append("")

    if plan.ready_for_execution:
        lines.append(
            "The plan is structurally complete and may be passed to an executor."
        )
    else:
        lines.append(
            "The planner retained the known reasoning structure, but no grid "
            "has been changed until every required value is resolved."
        )

    return "\n".join(lines)


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

    _, _, plan = learn_validate_and_plan_grid_pairs([
        (input_1, output_1),
        (input_2, output_2),
    ])

    print(format_rule_plan(plan))


if __name__ == "__main__":
    _self_test()