"""
Полная реализация алгоритма Форчуна (сканирующая прямая, пляжная линия, DCEL).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple, Optional

_FEPS = 1e-10


# --- Site / Vector -----------------------------------------------------------

@dataclass
class Site:
    x: float
    y: float
    idx: int = -1

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Site):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    @property
    def vector(self) -> "Vector2D":
        return Vector2D(dx=self.x, dy=self.y)

    def distance_to(self, other: "Site") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Vector2D:
    dx: float = 0.0
    dy: float = 0.0

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(dx=self.dx + other.dx, dy=self.dy + other.dy)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(dx=self.dx - other.dx, dy=self.dy - other.dy)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(dx=self.dx * scalar, dy=self.dy * scalar)

    __rmul__ = __mul__

    @property
    def normal(self) -> "Vector2D":
        return Vector2D(dx=-self.dy, dy=self.dx)

    @property
    def point(self) -> Site:
        return Site(x=self.dx, y=self.dy, idx=-1)


# --- Parabola / Circle -------------------------------------------------------

class Parabola:
    def __init__(self, focus: Site, directrix_y: float) -> None:
        self.focus = focus
        self.directrix_y = directrix_y

    def standard_form(self) -> tuple[float, float, float]:
        vertex_x = self.focus.x
        vertex_y = (self.focus.y + self.directrix_y) * 0.5
        parameter_p = self.focus.y - vertex_y
        if abs(parameter_p) < _FEPS:
            parameter_p = _FEPS if parameter_p >= 0 else -_FEPS
        coefficient_a = 1.0 / (4.0 * parameter_p)
        coefficient_b = -2.0 * vertex_x / (4.0 * parameter_p)
        coefficient_c = (vertex_x * vertex_x) / (4.0 * parameter_p) + vertex_y
        return coefficient_a, coefficient_b, coefficient_c

    def intersection_x(self, other: "Parabola") -> Optional[float]:
        focus_left = self.focus
        focus_right = other.focus
        directrix_y = self.directrix_y

        if focus_left.y == focus_right.y:
            return (focus_left.x + focus_right.x) * 0.5
        if focus_left.y == directrix_y:
            return focus_left.x
        if focus_right.y == directrix_y:
            return focus_right.x

        coefficient_a1, coefficient_b1, coefficient_c1 = self.standard_form()
        coefficient_a2, coefficient_b2, coefficient_c2 = other.standard_form()
        delta_a = coefficient_a1 - coefficient_a2
        delta_b = coefficient_b1 - coefficient_b2
        delta_c = coefficient_c1 - coefficient_c2

        if abs(delta_a) < _FEPS:
            if abs(delta_b) < _FEPS:
                return None
            return -delta_c / delta_b

        discriminant = delta_b * delta_b - 4.0 * delta_a * delta_c
        if discriminant < -_FEPS:
            return None
        discriminant = max(0.0, discriminant)
        sqrt_discriminant = math.sqrt(discriminant)

        root1 = (-delta_b + sqrt_discriminant) / (2.0 * delta_a)
        root2 = (-delta_b - sqrt_discriminant) / (2.0 * delta_a)

        return min(root1, root2) if focus_left.y < focus_right.y else max(root1, root2)


@dataclass
class Circle:
    center: Site
    radius: float

    @staticmethod
    def from_three_points(point1: Site, point2: Site, point3: Site) -> Optional["Circle"]:
        x1, y1 = point1.x, point1.y
        x2, y2 = point2.x, point2.y
        x3, y3 = point3.x, point3.y

        denominator = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(denominator) < _FEPS:
            return None

        norm1 = x1 * x1 + y1 * y1
        norm2 = x2 * x2 + y2 * y2
        norm3 = x3 * x3 + y3 * y3

        center_x = (norm1 * (y2 - y3) + norm2 * (y3 - y1) + norm3 * (y1 - y2)) / denominator
        center_y = (norm1 * (x3 - x2) + norm2 * (x1 - x3) + norm3 * (x2 - x1)) / denominator

        center = Site(x=center_x, y=center_y, idx=-2)
        radius = math.hypot(x1 - center_x, y1 - center_y)
        return Circle(center=center, radius=radius)

    @property
    def bottom_point(self) -> Site:
        return Site(x=self.center.x, y=self.center.y + self.radius, idx=-1)


# --- Line clip ---------------------------------------------------------------

@dataclass
class LineSegment:
    a: Site
    b: Site


class ClipperEdge(Enum):
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()


@dataclass
class Clipper:
    left: float
    right: float
    top: float
    bottom: float


class LiangBarskyResult(NamedTuple):
    is_origin_clipped: bool
    is_destination_clipped: bool
    result_segment: Optional[LineSegment]


def lb_clip(line: LineSegment, clipper: Clipper) -> LiangBarskyResult:
    parameter_start = 0.0
    parameter_end = 1.0
    delta_x = line.b.x - line.a.x
    delta_y = line.b.y - line.a.y
    is_origin_clipped = False
    is_destination_clipped = False

    for edge in ClipperEdge:
        if edge == ClipperEdge.LEFT:
            p_value, q_value = -delta_x, -(clipper.left - line.a.x)
        elif edge == ClipperEdge.RIGHT:
            p_value, q_value = delta_x, clipper.right - line.a.x
        elif edge == ClipperEdge.BOTTOM:
            p_value, q_value = -delta_y, -(clipper.bottom - line.a.y)
        else:
            p_value, q_value = delta_y, clipper.top - line.a.y

        if p_value == 0 and q_value < 0:
            return LiangBarskyResult(False, False, None)

        if p_value != 0:
            ratio = q_value / p_value
            if p_value < 0:
                if ratio > parameter_end:
                    return LiangBarskyResult(False, False, None)
                if ratio > parameter_start:
                    is_origin_clipped = True
                    parameter_start = ratio
            else:
                if ratio < parameter_start:
                    return LiangBarskyResult(False, False, None)
                if ratio < parameter_end:
                    is_destination_clipped = True
                    parameter_end = ratio

    clipped_segment = LineSegment(
        a=Site(
            x=line.a.x + parameter_start * delta_x,
            y=line.a.y + parameter_start * delta_y,
            idx=-1,
        ),
        b=Site(
            x=line.a.x + parameter_end * delta_x,
            y=line.a.y + parameter_end * delta_y,
            idx=-1,
        ),
    )
    return LiangBarskyResult(is_origin_clipped, is_destination_clipped, clipped_segment)


# --- Rectangle ---------------------------------------------------------------

class RectangleEdge(Enum):
    TOP = auto()
    RIGHT = auto()
    LEFT = auto()
    BOTTOM = auto()


@dataclass
class Rectangle:
    x: float
    y: float
    width: float
    height: float

    @property
    def tl(self) -> Site:
        return Site(x=self.x, y=self.y, idx=-1)

    @property
    def bl(self) -> Site:
        return Site(x=self.tl.x, y=self.tl.y + self.height, idx=-1)

    @property
    def tr(self) -> Site:
        return Site(x=self.tl.x + self.width, y=self.tl.y, idx=-1)

    @property
    def br(self) -> Site:
        return Site(x=self.tl.x + self.width, y=self.tl.y + self.height, idx=-1)

    @property
    def origin(self) -> Site:
        return self.tl

    def expand_to_contain_point(self, point: Site, padding: float = 20.0) -> None:
        if point.x <= self.origin.x:
            self.width += abs(self.x - point.x + padding)
            self.x = point.x - padding
        if point.y <= self.origin.y:
            self.height += abs(self.y - point.y + padding)
            self.y = point.y - padding
        if point.x >= self.origin.x + self.width:
            self.width = point.x - self.x + padding
        if point.y >= self.origin.y + self.height:
            self.height = point.y - self.y + padding

    def point_is_on_boundary(self, point: Optional[Site]) -> bool:
        if point is None:
            return False
        return self.side_for_point(point) is not None

    def get_line(self, edge: RectangleEdge) -> LineSegment:
        if edge == RectangleEdge.TOP:
            return LineSegment(a=self.tr, b=self.tl)
        if edge == RectangleEdge.RIGHT:
            return LineSegment(a=self.br, b=self.tr)
        if edge == RectangleEdge.BOTTOM:
            return LineSegment(a=self.bl, b=self.br)
        return LineSegment(a=self.tl, b=self.bl)

    @classmethod
    def rect_from_source(cls, source: "Rectangle", padding: float) -> "Rectangle":
        return cls(
            x=source.tl.x - padding,
            y=source.tl.y - padding,
            width=source.width + 2 * padding,
            height=source.height + 2 * padding,
        )

    def intersection(self, origin: Site, direction: Vector2D) -> tuple[Site, RectangleEdge]:
        candidates: list[tuple[float, Site, RectangleEdge]] = []

        if abs(direction.dx) > _FEPS:
            parameter_left = (self.tl.x - origin.x) / direction.dx
            if parameter_left >= -_FEPS:
                point_y = origin.y + parameter_left * direction.dy
                if self.tl.y - _FEPS <= point_y <= self.bl.y + _FEPS:
                    candidates.append((
                        max(parameter_left, 0.0),
                        Site(self.tl.x, min(max(point_y, self.tl.y), self.bl.y), -1),
                        RectangleEdge.LEFT,
                    ))

            parameter_right = (self.tr.x - origin.x) / direction.dx
            if parameter_right >= -_FEPS:
                point_y = origin.y + parameter_right * direction.dy
                if self.tr.y - _FEPS <= point_y <= self.br.y + _FEPS:
                    candidates.append((
                        max(parameter_right, 0.0),
                        Site(self.tr.x, min(max(point_y, self.tr.y), self.br.y), -1),
                        RectangleEdge.RIGHT,
                    ))

        if abs(direction.dy) > _FEPS:
            parameter_top = (self.tl.y - origin.y) / direction.dy
            if parameter_top >= -_FEPS:
                point_x = origin.x + parameter_top * direction.dx
                if self.tl.x - _FEPS <= point_x <= self.tr.x + _FEPS:
                    candidates.append((
                        max(parameter_top, 0.0),
                        Site(min(max(point_x, self.tl.x), self.tr.x), self.tl.y, -1),
                        RectangleEdge.TOP,
                    ))

            parameter_bottom = (self.bl.y - origin.y) / direction.dy
            if parameter_bottom >= -_FEPS:
                point_x = origin.x + parameter_bottom * direction.dx
                if self.bl.x - _FEPS <= point_x <= self.br.x + _FEPS:
                    candidates.append((
                        max(parameter_bottom, 0.0),
                        Site(min(max(point_x, self.bl.x), self.br.x), self.bl.y, -1),
                        RectangleEdge.BOTTOM,
                    ))

        if not candidates:
            raise ValueError("Луч не пересекает прямоугольник")

        unique_candidates: list[tuple[float, Site, RectangleEdge]] = []
        for current_parameter, current_point, current_edge in candidates:
            is_duplicate = False
            for _, existing_point, _ in unique_candidates:
                if (
                    abs(current_point.x - existing_point.x) < _FEPS
                    and abs(current_point.y - existing_point.y) < _FEPS
                ):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_candidates.append((current_parameter, current_point, current_edge))

        unique_candidates.sort(key=lambda item: item[0])
        _, point, edge = unique_candidates[0]
        return point, edge

    def _get_next_ccw(self, edge: RectangleEdge) -> tuple[RectangleEdge, Site]:
        if edge == RectangleEdge.LEFT:
            return RectangleEdge.BOTTOM, self.bl
        if edge == RectangleEdge.BOTTOM:
            return RectangleEdge.RIGHT, self.br
        if edge == RectangleEdge.RIGHT:
            return RectangleEdge.TOP, self.tr
        return RectangleEdge.LEFT, self.tl

    def side_for_point(self, point: Site) -> Optional[RectangleEdge]:
        for edge in (
            RectangleEdge.TOP,
            RectangleEdge.RIGHT,
            RectangleEdge.BOTTOM,
            RectangleEdge.LEFT,
        ):
            if self.get_line(edge).contains_point(point):
                return edge
        return None

    def ccw_traverse(self, start_edge: RectangleEdge, end_edge: RectangleEdge) -> list[Site]:
        points: list[Site] = []
        edge = start_edge
        while edge != end_edge:
            next_edge, corner = self._get_next_ccw(edge)
            edge = next_edge
            points.append(corner)
        return points

    def get_rect_polyline_for_ccw(self, start: Site, end: Site) -> list[Site]:
        result: list[Site] = []
        start_edge = self.side_for_point(start)
        end_edge = self.side_for_point(end)

        if start_edge is None or end_edge is None:
            return result

        if start_edge == end_edge:
            segment = self.get_line(start_edge)
            delta_x = segment.b.x - segment.a.x
            delta_y = segment.b.y - segment.a.y
            segment_length_squared = delta_x * delta_x + delta_y * delta_y

            if segment_length_squared < _FEPS:
                return result

            start_parameter = (
                (start.x - segment.a.x) * delta_x + (start.y - segment.a.y) * delta_y
            ) / segment_length_squared
            end_parameter = (
                (end.x - segment.a.x) * delta_x + (end.y - segment.a.y) * delta_y
            ) / segment_length_squared

            if end_parameter >= start_parameter - _FEPS:
                return []

            next_edge, corner = self._get_next_ccw(start_edge)
            result.append(corner)
            result.extend(self.ccw_traverse(next_edge, start_edge))
            return result

        result.extend(self.ccw_traverse(start_edge, end_edge))
        return result

    def to_clipper(self) -> Clipper:
        return Clipper(left=self.tl.x, right=self.tr.x, top=self.tr.y, bottom=self.br.y)


def _segment_contains_point(segment: LineSegment, point: Site) -> bool:
    cross_value = (
        (segment.b.x - segment.a.x) * (point.y - segment.a.y)
        - (segment.b.y - segment.a.y) * (point.x - segment.a.x)
    )
    if abs(cross_value) > _FEPS:
        return False

    return (
        min(segment.a.x, segment.b.x) - _FEPS <= point.x <= max(segment.a.x, segment.b.x) + _FEPS
        and min(segment.a.y, segment.b.y) - _FEPS <= point.y <= max(segment.a.y, segment.b.y) + _FEPS
    )


LineSegment.contains_point = _segment_contains_point  # type: ignore[attr-defined]


def _site_distance_to(first_point: Site, second_point: Site) -> float:
    return math.hypot(first_point.x - second_point.x, first_point.y - second_point.y)


def _line_segment_distance_to(self: LineSegment, point: Site) -> float:
    return _site_distance_to(self.a, point)


LineSegment.distance_to = _line_segment_distance_to  # type: ignore[attr-defined]


# --- DCEL --------------------------------------------------------------------

@dataclass
class HalfEdge:
    origin: Optional[Site] = None
    destination: Optional[Site] = None
    twin: Optional["HalfEdge"] = None
    prev: Optional["HalfEdge"] = None
    next: Optional["HalfEdge"] = None
    incident_face: Optional["Cell"] = None


@dataclass
class Cell:
    site: Site
    outer_component: Optional[HalfEdge] = None


@dataclass
class Diagram:
    cells: list[Cell] = field(default_factory=list)
    vertices: list[Site] = field(default_factory=list)

    def create_cell(self, arc: "Arc") -> None:
        cell = Cell(site=arc.point)
        self.cells.append(cell)
        arc.cell = cell

    def create_half_edge(self, cell: Cell) -> HalfEdge:
        half_edge = HalfEdge(incident_face=cell)
        if cell.outer_component is None:
            cell.outer_component = half_edge
        return half_edge

    def clear(self) -> None:
        self.cells.clear()
        self.vertices.clear()


# --- Beachline (RB-tree) -----------------------------------------------------

class Arc:
    def __init__(self, point: Optional[Site] = None) -> None:
        self.is_black = True
        self.right: Optional["Arc"] = None
        self.left: Optional["Arc"] = None
        self.parent: Optional["Arc"] = None
        self.point = point
        self.circle_eid: int = -1
        self.prev: Optional["Arc"] = None
        self.next: Optional["Arc"] = None
        self.left_half_edge: Optional[HalfEdge] = None
        self.right_half_edge: Optional[HalfEdge] = None
        self.cell: Optional[Cell] = None

    def bounds(self, directrix_y: float) -> tuple[float, float]:
        left_bound = float("-inf")
        right_bound = float("inf")
        parabola = Parabola(focus=self.point, directrix_y=directrix_y)  # type: ignore[arg-type]

        if self.prev:
            left_parabola = Parabola(focus=self.prev.point, directrix_y=directrix_y)  # type: ignore[arg-type]
            intersection_x = left_parabola.intersection_x(parabola)
            if intersection_x is not None:
                left_bound = intersection_x

        if self.next:
            right_parabola = Parabola(focus=self.next.point, directrix_y=directrix_y)  # type: ignore[arg-type]
            intersection_x = parabola.intersection_x(right_parabola)
            if intersection_x is not None:
                right_bound = intersection_x

        return left_bound, right_bound


class Beachline:
    def __init__(self) -> None:
        self.sweepline_y = 0.0
        self.sentinel = Arc()
        self.root: Optional[Arc] = None

    def _minimum(self, node: Arc) -> Arc:
        while node.left is not self.sentinel:
            node = node.left  # type: ignore[assignment]
        return node

    def _maximum(self, node: Arc) -> Arc:
        while node.right is not self.sentinel:
            node = node.right  # type: ignore[assignment]
        return node

    def _transplant(self, node_u: Arc, node_v: Arc) -> None:
        if node_u.parent is self.sentinel:
            self.root = node_v
        elif node_u is node_u.parent.left:
            node_u.parent.left = node_v
        else:
            node_u.parent.right = node_v
        node_v.parent = node_u.parent

    def _left_rotate(self, node_x: Arc) -> None:
        node_y = node_x.right  # type: ignore[assignment]
        node_x.right = node_y.left
        if node_y.left is not self.sentinel:
            node_y.left.parent = node_x
        node_y.parent = node_x.parent
        if node_x.parent is self.sentinel:
            self.root = node_y
        elif node_x is node_x.parent.left:
            node_x.parent.left = node_y
        else:
            node_x.parent.right = node_y
        node_y.left = node_x
        node_x.parent = node_y

    def _right_rotate(self, node_x: Arc) -> None:
        node_y = node_x.left  # type: ignore[assignment]
        node_x.left = node_y.right
        if node_y.right is not self.sentinel:
            node_y.right.parent = node_x
        node_y.parent = node_x.parent
        if node_x.parent is self.sentinel:
            self.root = node_y
        elif node_x is node_x.parent.right:
            node_x.parent.right = node_y
        else:
            node_x.parent.left = node_y
        node_y.right = node_x
        node_x.parent = node_y

    def insert_fixup(self, node_z: Arc) -> None:
        while node_z.parent is not self.sentinel and not node_z.parent.is_black:
            grandparent = node_z.parent.parent
            if grandparent is self.sentinel:
                break
            if node_z.parent is grandparent.left:
                node_y = grandparent.right
                if node_y is not self.sentinel and not node_y.is_black:
                    node_z.parent.is_black = True
                    node_y.is_black = True
                    grandparent.is_black = False
                    node_z = grandparent
                else:
                    if node_z is node_z.parent.right:
                        node_z = node_z.parent
                        self._left_rotate(node_z)
                    node_z.parent.is_black = True
                    grandparent.is_black = False
                    self._right_rotate(grandparent)
            else:
                node_y = grandparent.left
                if node_y is not self.sentinel and not node_y.is_black:
                    node_z.parent.is_black = True
                    node_y.is_black = True
                    grandparent.is_black = False
                    node_z = grandparent
                else:
                    if node_z is node_z.parent.left:
                        node_z = node_z.parent
                        self._right_rotate(node_z)
                    node_z.parent.is_black = True
                    grandparent.is_black = False
                    self._left_rotate(grandparent)
        if self.root:
            self.root.is_black = True

    def delete_fixup(self, node_x: Arc) -> None:
        while node_x is not self.root and node_x.is_black:
            if node_x.parent and node_x is node_x.parent.left:
                node_w = node_x.parent.right
                if node_w and not node_w.is_black:
                    node_w.is_black = True
                    node_x.parent.is_black = False
                    self._left_rotate(node_x.parent)
                    node_w = node_x.parent.right
                if (
                    node_w
                    and node_w.left
                    and node_w.right
                    and node_w.left.is_black
                    and node_w.right.is_black
                ):
                    node_w.is_black = False
                    node_x = node_x.parent
                else:
                    if node_w and node_w.right and node_w.right.is_black:
                        if node_w.left:
                            node_w.left.is_black = True
                        node_w.is_black = False
                        self._right_rotate(node_w)
                        node_w = node_x.parent.right
                    if node_w and node_x.parent:
                        node_w.is_black = node_x.parent.is_black
                        node_x.parent.is_black = True
                        if node_w.right:
                            node_w.right.is_black = True
                        self._left_rotate(node_x.parent)
                    node_x = self.root  # type: ignore[assignment]
            else:
                node_w = node_x.parent.left if node_x.parent else None
                if node_w and not node_w.is_black:
                    node_w.is_black = True
                    if node_x.parent:
                        node_x.parent.is_black = False
                        self._right_rotate(node_x.parent)
                    node_w = node_x.parent.left if node_x.parent else None
                if (
                    node_w
                    and node_w.right
                    and node_w.left
                    and node_w.right.is_black
                    and node_w.left.is_black
                ):
                    node_w.is_black = False
                    node_x = node_x.parent
                else:
                    if node_w and node_w.left and node_w.left.is_black:
                        if node_w.right:
                            node_w.right.is_black = True
                        node_w.is_black = False
                        self._left_rotate(node_w)
                        node_w = node_x.parent.left if node_x.parent else None
                    if node_w and node_x.parent:
                        node_w.is_black = node_x.parent.is_black
                        node_x.parent.is_black = True
                        if node_w.left:
                            node_w.left.is_black = True
                        self._right_rotate(node_x.parent)
                    node_x = self.root  # type: ignore[assignment]
        node_x.is_black = True

    def delete(self, node_z: Arc) -> None:
        node_y = node_z
        original_color = node_y.is_black

        if node_z.left is self.sentinel:
            node_x = node_z.right  # type: ignore[assignment]
            self._transplant(node_z, node_z.right)
        elif node_z.right is self.sentinel:
            node_x = node_z.left  # type: ignore[assignment]
            self._transplant(node_z, node_z.left)
        else:
            node_y = self._minimum(node_z.right)  # type: ignore[arg-type]
            original_color = node_y.is_black
            node_x = node_y.right  # type: ignore[assignment]
            if node_y.parent is node_z:
                node_x.parent = node_y
            else:
                self._transplant(node_y, node_y.right)
                node_y.right = node_z.right
                node_y.right.parent = node_y
            self._transplant(node_z, node_y)
            node_y.left = node_z.left
            node_y.left.parent = node_y
            node_y.is_black = node_z.is_black

        if original_color:
            self.delete_fixup(node_x)

    @property
    def is_empty(self) -> bool:
        return self.root is None

    def insert_root_arc(self, point: Site) -> Arc:
        self.root = Arc(point=point)
        self.root.left = self.sentinel
        self.root.right = self.sentinel
        self.root.parent = self.sentinel
        self.root.is_black = True
        return self.root

    def update_sweepline_y(self, y_value: float) -> None:
        self.sweepline_y = y_value

    def add_as_left_child(self, node_x: Arc, node_y: Arc) -> None:
        node_y.left = node_x
        node_x.parent = node_y
        node_x.left = self.sentinel
        node_x.right = self.sentinel
        node_x.is_black = False
        self.insert_fixup(node_x)

    def add_as_right_child(self, node_x: Arc, node_y: Arc) -> None:
        node_y.right = node_x
        node_x.parent = node_y
        node_x.left = self.sentinel
        node_x.right = self.sentinel
        node_x.is_black = False
        self.insert_fixup(node_x)

    def insert_arc_for_point(self, point: Site) -> tuple[Arc, bool]:
        middle_arc = Arc(point=point)
        current_arc = self.root
        assert current_arc is not None
        is_edge_case = False

        while True:
            assert current_arc.point is not None
            left_bound, right_bound = current_arc.bounds(self.sweepline_y)

            if point.x < left_bound:
                current_arc = current_arc.left  # type: ignore[assignment]
            elif point.x > right_bound:
                current_arc = current_arc.right  # type: ignore[assignment]
            elif abs(point.x - left_bound) < _FEPS:
                self.insert_successor(current_arc.prev, middle_arc)  # type: ignore[arg-type]
                is_edge_case = True
                break
            elif abs(point.x - right_bound) < _FEPS:
                self.insert_successor(current_arc, middle_arc)
                is_edge_case = True
                break
            else:
                self.insert_successor(current_arc, middle_arc)
                right_arc_copy = Arc(point=current_arc.point)
                self.insert_successor(middle_arc, right_arc_copy)
                is_edge_case = False
                break

        return middle_arc, is_edge_case

    def handle_special_arc_insertion_case(self, point: Site) -> Arc:
        arc = Arc(point=point)
        current_arc = self.root
        assert current_arc is not None
        while current_arc.next is not None:
            current_arc = current_arc.next
        self.insert_successor(current_arc, arc)
        return arc

    def insert_successor(self, previous_arc: Arc, new_arc: Arc) -> None:
        new_arc.prev = previous_arc
        new_arc.next = previous_arc.next
        previous_arc.next = new_arc
        if new_arc.next:
            new_arc.next.prev = new_arc

        if previous_arc.right is self.sentinel:
            self.add_as_right_child(new_arc, previous_arc)
        else:
            right_subtree = previous_arc.right
            while right_subtree.left is not self.sentinel:
                right_subtree = right_subtree.left
            self.add_as_left_child(new_arc, right_subtree)

    def delete_arc(self, arc: Arc) -> None:
        previous_arc = arc.prev
        next_arc = arc.next
        if previous_arc:
            previous_arc.next = next_arc
        if next_arc:
            next_arc.prev = previous_arc
        self.delete(arc)

    @property
    def minimum(self) -> Optional[Arc]:
        if self.root is None or self.root is self.sentinel:
            return None
        return self._minimum(self.root)

    @property
    def maximum(self) -> Optional[Arc]:
        if self.root is None or self.root is self.sentinel:
            return None
        return self._maximum(self.root)


# --- Fortune driver ----------------------------------------------------------

class FortuneVoronoi:
    @dataclass(order=True)
    class QueueEvent:
        y: float
        x: float
        seq: int
        kind: str = field(compare=False)  # "site" | "circle"
        payload: object = field(compare=False)

    def __init__(self, sites_math: list[tuple[float, float]]) -> None:
        self.sites_math = sites_math
        self.internal_sites = [
            Site(x=point_x, y=-point_y, idx=index)
            for index, (point_x, point_y) in enumerate(sites_math)
        ]
        self.event_heap: list[FortuneVoronoi.QueueEvent] = []
        self._event_seq = 0
        self.beachline = Beachline()
        self.sweep_line_y = 0.0
        self.first_site_y: Optional[float] = None
        self.container: Optional[Rectangle] = None
        self.clipper_rect: Rectangle
        self.diagram = Diagram()

        x_values = [point[0] for point in sites_math]
        y_values = [point[1] for point in sites_math]
        padding = 0.15 * max(max(x_values) - min(x_values), max(y_values) - min(y_values), 1.0)

        min_x = min(x_values) - padding
        max_x = max(x_values) + padding
        min_y_internal = -max(y_values) - padding
        max_y_internal = -min(y_values) + padding

        self.clipper_rect = Rectangle(
            x=min_x,
            y=min_y_internal,
            width=max_x - min_x,
            height=max_y_internal - min_y_internal,
        )

    def _push_site(self, site: Site) -> None:
        self._event_seq += 1
        heapq.heappush(
            self.event_heap,
            self.QueueEvent(site.y, site.x, self._event_seq, "site", site),
        )

    def _push_circle(self, bottom_y: float, center_x: float, arc: Arc) -> None:
        self._event_seq += 1
        event_id = self._event_seq
        arc.circle_eid = event_id
        heapq.heappush(
            self.event_heap,
            self.QueueEvent(bottom_y, center_x, event_id, "circle", arc),
        )

    def _invalidate_circle(self, arc: Optional[Arc]) -> None:
        if arc is not None:
            arc.circle_eid = -1

    def make_twins(self, first_half_edge: HalfEdge, second_half_edge: HalfEdge) -> None:
        first_half_edge.twin = second_half_edge
        second_half_edge.twin = first_half_edge

    def connect(self, previous_half_edge: HalfEdge, next_half_edge: HalfEdge) -> None:
        previous_half_edge.next = next_half_edge
        next_half_edge.prev = previous_half_edge

    def check_circle_event(
        self,
        left_arc: Optional[Arc],
        middle_arc: Optional[Arc],
        right_arc: Optional[Arc],
    ) -> Optional[Circle]:
        if left_arc is None or middle_arc is None or right_arc is None:
            return None

        point_a, point_b, point_c = left_arc.point, middle_arc.point, right_arc.point
        assert point_a and point_b and point_c

        circle = Circle.from_three_points(point_a, point_b, point_c)
        if circle is None:
            return None

        determinant = (
            point_b.x * point_c.y + point_a.x * point_b.y + point_a.y * point_c.x
            - point_a.y * point_b.x - point_b.y * point_c.x - point_a.x * point_c.y
        )

        event_y = circle.center.y + circle.radius
        if event_y >= self.sweep_line_y and determinant > 0:
            for point in self.internal_sites:
                if point.idx in (point_a.idx, point_b.idx, point_c.idx):
                    continue
                if math.hypot(point.x - circle.center.x, point.y - circle.center.y) < circle.radius - 1e-8:
                    return None
            return circle

        return None

    def create_circle_event(self, arc: Optional[Arc]) -> None:
        if arc is None:
            return
        left_arc = arc.prev
        right_arc = arc.next
        circle = self.check_circle_event(left_arc, arc, right_arc)
        if circle:
            bottom_point = circle.bottom_point
            self._push_circle(bottom_point.y, bottom_point.x, arc)

    def _reference_for_missing_origin(self, half_edge: Optional[HalfEdge]) -> Optional[Site]:
        if half_edge is None:
            return None
        if half_edge.destination is not None:
            return half_edge.destination
        return half_edge.origin

    def _reference_for_missing_destination(self, half_edge: Optional[HalfEdge]) -> Optional[Site]:
        if half_edge is None:
            return None
        if half_edge.origin is not None:
            return half_edge.origin
        return half_edge.destination

    def process_site_event(self, site: Site) -> None:
        self.sweep_line_y = site.y
        self.beachline.update_sweepline_y(self.sweep_line_y)

        if self.beachline.is_empty:
            root_arc = self.beachline.insert_root_arc(site)
            self.first_site_y = site.y
            self.container = Rectangle.rect_from_source(self.clipper_rect, 50.0)
            self.container.expand_to_contain_point(site)
            self.diagram.create_cell(root_arc)
            return

        if self.first_site_y is not None and abs(site.y - self.first_site_y) < _FEPS:
            self.container.expand_to_contain_point(site)  # type: ignore[union-attr]
            arc = self.beachline.handle_special_arc_insertion_case(site)
            self.diagram.create_cell(arc)
            previous_arc = arc.prev
            assert previous_arc is not None

            y_value = site.y - 1e6
            midpoint = Site(x=(previous_arc.point.x + arc.point.x) * 0.5, y=y_value, idx=-1)

            previous_arc.right_half_edge = self.diagram.create_half_edge(previous_arc.cell)  # type: ignore[arg-type]
            previous_arc.right_half_edge.destination = midpoint

            arc.left_half_edge = self.diagram.create_half_edge(arc.cell)  # type: ignore[arg-type]
            arc.left_half_edge.origin = midpoint

            self.make_twins(previous_arc.right_half_edge, arc.left_half_edge)
            return

        assert self.container is not None
        self.container.expand_to_contain_point(site)

        new_arc, is_special = self.beachline.insert_arc_for_point(site)
        self.diagram.create_cell(new_arc)

        self._invalidate_circle(new_arc.prev)
        self.create_circle_event(new_arc.prev)
        self.create_circle_event(new_arc.next)

        next_arc = new_arc.next
        previous_arc = new_arc.prev
        assert previous_arc is not None and next_arc is not None

        if is_special:
            vertex_circle = Circle.from_three_points(previous_arc.point, new_arc.point, next_arc.point)  # type: ignore[arg-type]
            assert vertex_circle is not None
            vertex = vertex_circle.center

            self.container.expand_to_contain_point(vertex)
            self.diagram.vertices.append(vertex)

            previous_arc.right_half_edge.origin = vertex
            next_arc.left_half_edge.destination = vertex

            left_half_edge = self.diagram.create_half_edge(new_arc.cell)  # type: ignore[arg-type]
            new_arc.left_half_edge = left_half_edge
            left_half_edge.origin = vertex

            left_twin = self.diagram.create_half_edge(previous_arc.cell)  # type: ignore[arg-type]
            left_twin.destination = vertex
            self.make_twins(left_half_edge, left_twin)

            right_half_edge = self.diagram.create_half_edge(new_arc.cell)  # type: ignore[arg-type]
            new_arc.right_half_edge = right_half_edge
            right_half_edge.destination = vertex

            right_twin = self.diagram.create_half_edge(next_arc.cell)  # type: ignore[arg-type]
            right_twin.origin = vertex
            self.make_twins(right_half_edge, right_twin)

            self.connect(previous_arc.right_half_edge, left_half_edge)
            self.connect(right_half_edge, next_arc.left_half_edge)

            previous_arc.right_half_edge = left_twin
            next_arc.left_half_edge = right_twin
        else:
            next_arc.cell = previous_arc.cell
            next_arc.right_half_edge = previous_arc.right_half_edge

            previous_arc.right_half_edge = self.diagram.create_half_edge(previous_arc.cell)  # type: ignore[arg-type]
            new_arc.left_half_edge = self.diagram.create_half_edge(new_arc.cell)  # type: ignore[arg-type]

            self.make_twins(previous_arc.right_half_edge, new_arc.left_half_edge)

            new_arc.right_half_edge = new_arc.left_half_edge
            next_arc.left_half_edge = previous_arc.right_half_edge

    def _initialize_first_row(self, sorted_sites: list[Site]) -> int:
        if not sorted_sites:
            return 0

        first_y = sorted_sites[0].y
        count = 0
        while count < len(sorted_sites) and abs(sorted_sites[count].y - first_y) < _FEPS:
            count += 1

        initial_sites = sorted(sorted_sites[:count], key=lambda point: point.x)
        for site in initial_sites:
            self.process_site_event(site)

        return count

    def _delete_min(self) -> Optional["QueueEvent"]:
        if not self.event_heap:
            return None
        return heapq.heappop(self.event_heap)

    def create_vertex(self, vertex: Site, removed_arc: Arc) -> None:
        assert self.container is not None
        self.container.expand_to_contain_point(vertex)
        self.diagram.vertices.append(vertex)

        previous_arc = removed_arc.prev
        next_arc = removed_arc.next

        if removed_arc.left_half_edge:
            removed_arc.left_half_edge.destination = vertex
        if removed_arc.right_half_edge:
            removed_arc.right_half_edge.origin = vertex

        if previous_arc and previous_arc.right_half_edge:
            previous_arc.right_half_edge.origin = vertex
            if previous_arc.right_half_edge.twin:
                previous_arc.right_half_edge.twin.destination = vertex

        if next_arc and next_arc.left_half_edge:
            next_arc.left_half_edge.destination = vertex
            if next_arc.left_half_edge.twin:
                next_arc.left_half_edge.twin.origin = vertex

        if (
            previous_arc is not None
            and next_arc is not None
            and previous_arc.right_half_edge is not None
            and next_arc.left_half_edge is not None
            and previous_arc.right_half_edge.twin is not None
            and next_arc.left_half_edge.twin is not None
        ):
            self.connect(previous_arc.right_half_edge.twin, next_arc.left_half_edge.twin)

        if previous_arc:
            previous_right_half_edge = self.diagram.create_half_edge(previous_arc.cell)  # type: ignore[arg-type]
            previous_right_half_edge.destination = vertex
            self.connect(previous_right_half_edge, previous_arc.right_half_edge)  # type: ignore[arg-type]
            previous_arc.right_half_edge = previous_right_half_edge

        if next_arc:
            next_left_half_edge = self.diagram.create_half_edge(next_arc.cell)  # type: ignore[arg-type]
            next_left_half_edge.origin = vertex
            self.connect(next_arc.left_half_edge, next_left_half_edge)
            next_arc.left_half_edge = next_left_half_edge

        if previous_arc and next_arc:
            self.make_twins(previous_arc.right_half_edge, next_arc.left_half_edge)

    def process_circle_event(self, arc: Arc, event_id: int) -> None:
        if arc.circle_eid != event_id:
            return

        left_arc = arc.prev
        right_arc = arc.next
        if left_arc is None or right_arc is None:
            return

        circle = Circle.from_three_points(left_arc.point, arc.point, right_arc.point)  # type: ignore[arg-type]
        if circle is None:
            return

        center = circle.center
        self.sweep_line_y = center.y + circle.radius
        self.beachline.update_sweepline_y(self.sweep_line_y)

        self.beachline.delete_arc(arc)
        self._invalidate_circle(left_arc)
        self._invalidate_circle(arc)
        self._invalidate_circle(right_arc)

        self.create_vertex(center, arc)
        self.create_circle_event(left_arc)
        self.create_circle_event(right_arc)

    def get_box_intersection(
        self,
        point1: Site,
        point2: Site,
        rectangle: Rectangle,
        reference_point: Optional[Site] = None,
    ) -> Site:
        midpoint = ((point1.vector + point2.vector) * 0.5).point
        direction = (point2.vector - point1.vector).normal

        first_point, _ = rectangle.intersection(midpoint, direction)
        second_point, _ = rectangle.intersection(midpoint, direction * -1.0)

        if reference_point is not None:
            reference_vector_x = reference_point.x - midpoint.x
            reference_vector_y = reference_point.y - midpoint.y

            first_projection = (
                (first_point.x - midpoint.x) * reference_vector_x
                + (first_point.y - midpoint.y) * reference_vector_y
            )
            second_projection = (
                (second_point.x - midpoint.x) * reference_vector_x
                + (second_point.y - midpoint.y) * reference_vector_y
            )

            if abs(first_projection - second_projection) > _FEPS:
                if first_projection > second_projection:
                    return first_point
                return second_point

        first_distance = midpoint.distance_to(first_point)
        second_distance = midpoint.distance_to(second_point)

        if first_distance >= second_distance:
            return first_point
        return second_point

    def half_edges_chain(
        self,
        cell: Cell,
        clipping_rect: Rectangle,
        start: Site,
        end: Site,
    ) -> tuple[HalfEdge, HalfEdge]:
        points = clipping_rect.get_rect_polyline_for_ccw(start, end)
        head = self.diagram.create_half_edge(cell)
        head.origin = start
        current_half_edge = head

        if not points:
            current_half_edge.destination = end
            return head, current_half_edge

        for point in points:
            current_half_edge.destination = point
            new_half_edge = self.diagram.create_half_edge(cell)
            new_half_edge.origin = point
            self.connect(current_half_edge, new_half_edge)
            current_half_edge = new_half_edge

        current_half_edge.destination = end
        return head, current_half_edge

    def bound_incomplete_arc(self, arc: Arc) -> None:
        assert self.container is not None

        start_point: Optional[Site] = None
        end_point: Optional[Site] = None

        if arc.prev is not None and arc.prev.right_half_edge is not None:
            previous_reference_point = self._reference_for_missing_origin(
                arc.prev.right_half_edge
            )

            start_point = self.get_box_intersection(
                arc.prev.point,
                arc.point,
                self.container,
                previous_reference_point,
            )
            arc.prev.right_half_edge.origin = start_point

        if arc.next is not None and arc.next.left_half_edge is not None:
            next_reference_point = self._reference_for_missing_destination(
                arc.next.left_half_edge
            )

            end_point = self.get_box_intersection(
                arc.point,
                arc.next.point,
                self.container,
                next_reference_point,
            )
            arc.next.left_half_edge.destination = end_point

        if (
            start_point is not None
            and end_point is not None
            and arc.left_half_edge is not None
            and arc.right_half_edge is not None
        ):
            head, tail = self.half_edges_chain(
                arc.cell,
                self.container,
                start_point,
                end_point,
            )
            self.connect(arc.left_half_edge, head)
            self.connect(tail, arc.right_half_edge)

    def terminate(self) -> None:
        assert self.container is not None

        arc = self.beachline.minimum
        while arc:
            self.bound_incomplete_arc(arc)
            arc = arc.next

        minimum_arc = self.beachline.minimum
        maximum_arc = self.beachline.maximum

        if (
            minimum_arc
            and maximum_arc
            and minimum_arc.cell == maximum_arc.cell
            and minimum_arc is not maximum_arc
        ):
            previous_arc = maximum_arc.prev
            next_arc = minimum_arc.next

            if previous_arc and next_arc:
                left_reference_point = self._reference_for_missing_destination(
                    maximum_arc.left_half_edge
                )
                right_reference_point = self._reference_for_missing_origin(
                    minimum_arc.right_half_edge
                )

                maximum_arc.left_half_edge.destination = self.get_box_intersection(
                    previous_arc.point,
                    maximum_arc.point,
                    self.container,
                    left_reference_point,
                )

                minimum_arc.right_half_edge.origin = self.get_box_intersection(
                    minimum_arc.point,
                    next_arc.point,
                    self.container,
                    right_reference_point,
                )

                start = minimum_arc.right_half_edge.origin
                end = maximum_arc.left_half_edge.destination

                if start and end:
                    head, tail = self.half_edges_chain(
                        maximum_arc.cell,
                        self.container,
                        end,
                        start,
                    )
                    self.connect(maximum_arc.left_half_edge, head)
                    self.connect(tail, minimum_arc.right_half_edge)

    def run(self) -> Diagram:
        sorted_sites = sorted(self.internal_sites, key=lambda point: (point.y, point.x))
        start_index = self._initialize_first_row(sorted_sites)

        for site in sorted_sites[start_index:]:
            self._push_site(site)

        while self.event_heap:
            current_event = self._delete_min()
            if current_event is None:
                break

            self.sweep_line_y = current_event.y
            if current_event.kind == "site":
                self.process_site_event(current_event.payload)  # type: ignore[arg-type]
            else:
                self.process_circle_event(current_event.payload, current_event.seq)  # type: ignore[arg-type]

        self.terminate()
        return self.diagram


def diagram_to_voronoi_result(
    diagram: Diagram,
    site_count: int,
    clip_bbox_math: tuple[float, float, float, float],
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[int, int, Optional[tuple[float, float]], Optional[tuple[float, float]]]],
]:
    xmin, xmax, ymin, ymax = clip_bbox_math

    def to_math(site: Optional[Site]) -> Optional[tuple[float, float]]:
        if site is None:
            return None
        return (site.x, -site.y)

    output_sites = [(0.0, 0.0)] * site_count
    seen_edge_pairs: set[tuple[int, int]] = set()
    ridges: list[tuple[int, int, Optional[tuple[float, float]], Optional[tuple[float, float]]]] = []

    for cell in diagram.cells:
        site_index = cell.site.idx
        if 0 <= site_index < site_count:
            output_sites[site_index] = to_math(cell.site)  # type: ignore[assignment]

    def liang_clip(
        first_point: tuple[float, float],
        second_point: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        start_site = Site(x=first_point[0], y=first_point[1], idx=-1)
        end_site = Site(x=second_point[0], y=second_point[1], idx=-1)

        clipper = Clipper(
            left=xmin,
            right=xmax,
            top=ymax,
            bottom=ymin,
        )

        clip_result = lb_clip(LineSegment(a=start_site, b=end_site), clipper)

        if clip_result.result_segment is None:
            return first_point, second_point

        return (
            (clip_result.result_segment.a.x, clip_result.result_segment.a.y),
            (clip_result.result_segment.b.x, clip_result.result_segment.b.y),
        )

    for cell in diagram.cells:
        start_half_edge = cell.outer_component
        if start_half_edge is None:
            continue

        current_half_edge = start_half_edge
        for _ in range(max(4, len(diagram.cells) * 8)):
            twin_half_edge = current_half_edge.twin
            if twin_half_edge and twin_half_edge.incident_face:
                first_index = cell.site.idx
                second_index = twin_half_edge.incident_face.site.idx

                if (
                    first_index >= 0
                    and second_index >= 0
                    and first_index != second_index
                    and first_index < site_count
                    and second_index < site_count
                ):
                    pair = (first_index, second_index) if first_index < second_index else (second_index, first_index)
                    if pair not in seen_edge_pairs:
                        seen_edge_pairs.add(pair)

                        origin_point = to_math(current_half_edge.origin)
                        destination_point = to_math(current_half_edge.destination)

                        if origin_point and destination_point:
                            clipped_start, clipped_end = liang_clip(origin_point, destination_point)
                            if math.hypot(clipped_start[0] - clipped_end[0], clipped_start[1] - clipped_end[1]) > 1e-8:
                                ridges.append((pair[0], pair[1], clipped_start, clipped_end))

            if current_half_edge.next is None:
                break

            current_half_edge = current_half_edge.next
            if current_half_edge is start_half_edge:
                break

    vertices: list[tuple[float, float]] = []
    seen_vertices: set[tuple[int, int]] = set()

    for vertex in diagram.vertices:
        rounded_key = (round(vertex.x, 9), round(vertex.y, 9))
        if rounded_key not in seen_vertices:
            seen_vertices.add(rounded_key)
            math_vertex = to_math(vertex)
            if math_vertex:
                vertices.append(math_vertex)

    return output_sites, vertices, ridges


def build_fortune_voronoi(
    sites_math: list[tuple[float, float]],
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[int, int, Optional[tuple[float, float]], Optional[tuple[float, float]]]],
]:
    site_count = len(sites_math)
    if site_count == 0:
        return [], [], []
    if site_count == 1:
        return [sites_math[0]], [], []

    fortune_voronoi = FortuneVoronoi(sites_math)
    diagram = fortune_voronoi.run()
    bbox = fortune_voronoi.clipper_rect

    clip_bbox_math = (bbox.tl.x, bbox.br.x, -bbox.br.y, -bbox.tl.y)
    return diagram_to_voronoi_result(diagram, site_count, clip_bbox_math)