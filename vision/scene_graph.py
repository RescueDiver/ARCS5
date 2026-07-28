from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from vision.objects import (
    ObjectDetectionResult,
    VisualObject,
    detect_objects,
)
from vision.relationships import (
    ObjectRelationship,
    RelationshipResult,
    detect_relationships,
)
from vision.scene_classifier import (
    SceneClassificationResult,
    SceneElement,
    SceneRole,
    classify_scene,
)


# =============================================================================
# GRAPH DATA TYPES
# =============================================================================

@dataclass(frozen=True)
class SceneGraphNode:
    object_id: int
    color: int
    cell_count: int
    cells: tuple[tuple[int, int], ...]

    bbox_top: int
    bbox_left: int
    bbox_bottom: int
    bbox_right: int
    bbox_height: int
    bbox_width: int

    touches_border: bool
    fill_fraction: float
    is_solid_rectangle: bool
    is_horizontal_line: bool
    is_vertical_line: bool

    role: SceneRole
    role_confidence: float
    role_reasons: tuple[str, ...]

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (
            self.bbox_top,
            self.bbox_left,
            self.bbox_bottom,
            self.bbox_right,
        )

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox_top + self.bbox_bottom) / 2.0,
            (self.bbox_left + self.bbox_right) / 2.0,
        )

    @property
    def shape_name(self) -> str:
        if self.is_horizontal_line:
            return "horizontal_line"

        if self.is_vertical_line:
            return "vertical_line"

        if self.is_solid_rectangle:
            if self.bbox_height == self.bbox_width:
                return "solid_square"

            return "solid_rectangle"

        if self.cell_count == 1:
            return "single_cell"

        return "irregular"


@dataclass(frozen=True)
class SceneGraphEdge:
    source_id: int
    target_id: int
    relationship: str
    directed: bool = True

    @property
    def key(self) -> tuple[int, str, int]:
        return (
            self.source_id,
            self.relationship,
            self.target_id,
        )


@dataclass(frozen=True)
class SceneGraph:
    height: int
    width: int
    connectivity: int
    nodes: tuple[SceneGraphNode, ...]
    edges: tuple[SceneGraphEdge, ...]

    def node(self, object_id: int) -> SceneGraphNode | None:
        for node in self.nodes:
            if node.object_id == object_id:
                return node

        return None

    def outgoing_edges(
        self,
        object_id: int,
        relationship: str | None = None,
    ) -> tuple[SceneGraphEdge, ...]:
        return tuple(
            edge
            for edge in self.edges
            if edge.source_id == object_id
            and (
                relationship is None
                or edge.relationship == relationship
            )
        )

    def incoming_edges(
        self,
        object_id: int,
        relationship: str | None = None,
    ) -> tuple[SceneGraphEdge, ...]:
        return tuple(
            edge
            for edge in self.edges
            if edge.target_id == object_id
            and (
                relationship is None
                or edge.relationship == relationship
            )
        )

    def neighbors(
        self,
        object_id: int,
        relationship: str | None = None,
    ) -> tuple[int, ...]:
        neighbor_ids: set[int] = set()

        for edge in self.edges:
            if relationship is not None and edge.relationship != relationship:
                continue

            if edge.source_id == object_id:
                neighbor_ids.add(edge.target_id)

            if not edge.directed and edge.target_id == object_id:
                neighbor_ids.add(edge.source_id)

        return tuple(sorted(neighbor_ids))

    def nodes_with_role(
        self,
        role: SceneRole,
    ) -> tuple[SceneGraphNode, ...]:
        return tuple(
            node
            for node in self.nodes
            if node.role == role
        )

    def relationship_exists(
        self,
        source_id: int,
        relationship: str,
        target_id: int,
    ) -> bool:
        return any(
            edge.source_id == source_id
            and edge.relationship == relationship
            and edge.target_id == target_id
            for edge in self.edges
        )

    @property
    def background_nodes(self) -> tuple[SceneGraphNode, ...]:
        return self.nodes_with_role(SceneRole.BACKGROUND)

    @property
    def foreground_nodes(self) -> tuple[SceneGraphNode, ...]:
        return self.nodes_with_role(SceneRole.FOREGROUND)

    @property
    def divider_nodes(self) -> tuple[SceneGraphNode, ...]:
        return self.nodes_with_role(SceneRole.DIVIDER)

    @property
    def frame_nodes(self) -> tuple[SceneGraphNode, ...]:
        return self.nodes_with_role(SceneRole.FRAME)

    @property
    def void_nodes(self) -> tuple[SceneGraphNode, ...]:
        return self.nodes_with_role(SceneRole.VOID)


# =============================================================================
# HELPERS
# =============================================================================

UNDIRECTED_RELATIONSHIPS = frozenset({
    "touching",
    "edge_touching",
    "corner_touching",
    "overlaps",
    "same_color",
    "same_cell_count",
    "same_dimensions",
    "same_shape",
    "aligned_row",
    "aligned_column",
    "nearest",
})


def _grid_shape(
    grid: Sequence[Sequence[int]],
) -> tuple[int, int]:
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    return height, width


def _role_element_for_object(
    classification: SceneClassificationResult,
    object_id: int,
) -> SceneElement | None:
    return classification.element_for_object(object_id)


def _make_node(
    obj: VisualObject,
    role_element: SceneElement | None,
) -> SceneGraphNode:
    bbox = obj.bbox

    if role_element is None:
        role = SceneRole.UNCERTAIN
        confidence = 0.0
        reasons = ("scene role unavailable",)
    else:
        role = role_element.role
        confidence = role_element.confidence
        reasons = role_element.reasons

    return SceneGraphNode(
        object_id=obj.object_id,
        color=obj.color,
        cell_count=obj.cell_count,
        cells=tuple(sorted(obj.cells)),
        bbox_top=bbox.top,
        bbox_left=bbox.left,
        bbox_bottom=bbox.bottom,
        bbox_right=bbox.right,
        bbox_height=bbox.height,
        bbox_width=bbox.width,
        touches_border=obj.touches_border,
        fill_fraction=obj.fill_fraction,
        is_solid_rectangle=obj.is_solid_rectangle,
        is_horizontal_line=obj.is_horizontal_line,
        is_vertical_line=obj.is_vertical_line,
        role=role,
        role_confidence=confidence,
        role_reasons=reasons,
    )


def _relationship_records(
    relationships: RelationshipResult,
) -> tuple[ObjectRelationship, ...]:
    return tuple(relationships.relationships)


def _make_edges(
    relationships: RelationshipResult,
) -> tuple[SceneGraphEdge, ...]:
    """
    Convert every active Boolean fact in ObjectRelationship into an edge.

    One source/target record may therefore produce several graph edges, such as:

        touching
        edge_touching
        nearest
        left_of
    """
    edges: list[SceneGraphEdge] = []
    seen: set[tuple[int, str, int]] = set()

    for relationship in _relationship_records(relationships):
        source_id = relationship.source_object_id
        target_id = relationship.target_object_id

        for name in relationship.active_names():
            key = (source_id, name, target_id)

            if key in seen:
                continue

            seen.add(key)

            edges.append(
                SceneGraphEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relationship=name,
                    directed=name not in UNDIRECTED_RELATIONSHIPS,
                )
            )

    edges.sort(
        key=lambda edge: (
            edge.source_id,
            edge.relationship,
            edge.target_id,
        )
    )

    return tuple(edges)


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_scene_graph(
    grid: Sequence[Sequence[int]],
    detection: ObjectDetectionResult | None = None,
    relationships: RelationshipResult | None = None,
    classification: SceneClassificationResult | None = None,
    *,
    connectivity: int = 4,
    preferred_void_colors: Iterable[int] = (0,),
    background_hint: int | None = None,
) -> SceneGraph:
    """
    Build one structured graph representation of an ARC grid.

    This combines three already-independent layers:

        objects
        relationships
        scene roles

    It performs no transformation learning and no input/output comparison.
    """
    height, width = _grid_shape(grid)

    if detection is None:
        detection = detect_objects(
            grid,
            connectivity=connectivity,
        )

    if relationships is None:
        relationships = detect_relationships(detection)

    if classification is None:
        classification = classify_scene(
            grid,
            detection,
            connectivity=connectivity,
            preferred_void_colors=preferred_void_colors,
            background_hint=background_hint,
        )

    nodes = tuple(
        _make_node(
            obj,
            _role_element_for_object(
                classification,
                obj.object_id,
            ),
        )
        for obj in detection.objects
    )

    edges = _make_edges(relationships)

    return SceneGraph(
        height=height,
        width=width,
        connectivity=connectivity,
        nodes=nodes,
        edges=edges,
    )


# =============================================================================
# FORMATTING
# =============================================================================

def summarize_scene_graph(
    graph: SceneGraph,
) -> dict[str, object]:
    role_counts: dict[str, int] = {}
    relationship_counts: dict[str, int] = {}

    for node in graph.nodes:
        role_name = node.role.value
        role_counts[role_name] = role_counts.get(role_name, 0) + 1

    for edge in graph.edges:
        relationship_counts[edge.relationship] = (
            relationship_counts.get(edge.relationship, 0) + 1
        )

    return {
        "shape": (graph.height, graph.width),
        "connectivity": graph.connectivity,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "role_counts": role_counts,
        "relationship_counts": relationship_counts,
    }


def format_scene_graph(
    graph: SceneGraph,
) -> str:
    lines = [
        "=" * 72,
        "SCENE GRAPH",
        "=" * 72,
        f"Grid: {graph.height}x{graph.width}",
        f"Connectivity: {graph.connectivity}",
        f"Nodes: {len(graph.nodes)}",
        f"Edges: {len(graph.edges)}",
        "",
    ]

    for node in graph.nodes:
        lines.extend([
            f"Object {node.object_id}",
            f"  role: {node.role.value}",
            f"  confidence: {node.role_confidence:.3f}",
            f"  color: {node.color}",
            f"  cells: {node.cell_count}",
            (
                "  bbox: "
                f"top={node.bbox_top}, "
                f"left={node.bbox_left}, "
                f"h={node.bbox_height}, "
                f"w={node.bbox_width}"
            ),
            f"  shape: {node.shape_name}",
            f"  border: {'yes' if node.touches_border else 'no'}",
            "  relationships:",
        ])

        outgoing = graph.outgoing_edges(node.object_id)

        if not outgoing:
            lines.append("    none")
        else:
            for edge in outgoing:
                lines.append(
                    f"    {edge.relationship} -> Object {edge.target_id}"
                )

        lines.append("")

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

    graph = build_scene_graph(
        grid,
        connectivity=4,
    )

    print(format_scene_graph(graph))
    print("Summary:")
    print(summarize_scene_graph(graph))


if __name__ == "__main__":
    _self_test()