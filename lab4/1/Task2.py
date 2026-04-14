from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from _fortune_dcel import build_fortune_voronoi as build_fortune_voronoi_raw

Point = tuple[float, float]
Ridge = tuple[int, int, Optional[Point], Optional[Point]]

EPS = 1e-9

def orient(first_point: Point, second_point: Point, third_point: Point) -> float:
    return (
        (second_point[0] - first_point[0]) * (third_point[1] - first_point[1])
        - (second_point[1] - first_point[1]) * (third_point[0] - first_point[0])
    )

@dataclass
class VoronoiDiagramResult:
    sites: list[Point]
    vertices: list[Point]
    ridges: list[Ridge]

def squared_distance(first_point: Point, second_point: Point) -> float:
    delta_x = first_point[0] - second_point[0]
    delta_y = first_point[1] - second_point[1]
    return delta_x * delta_x + delta_y * delta_y

def build_points_bbox(
    points: list[Point],
    padding_ratio: float = 0.12,
) -> tuple[float, float, float, float]:
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    return (
        min_x - padding_ratio * width,
        max_x + padding_ratio * width,
        min_y - padding_ratio * height,
        max_y + padding_ratio * height,
    )


def ray_to_bbox_boundary(
    ray_origin: Point,
    ray_direction: Point,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> Point:
    origin_x, origin_y = ray_origin
    direction_x, direction_y = ray_direction
    best_parameter = float("inf")

    if abs(direction_x) > EPS:
        for border_x in (min_x, max_x):
            parameter = (border_x - origin_x) / direction_x
            if parameter >= -EPS:
                current_y = origin_y + parameter * direction_y
                if min_y - EPS <= current_y <= max_y + EPS:
                    best_parameter = min(best_parameter, max(parameter, 0.0))

    if abs(direction_y) > EPS:
        for border_y in (min_y, max_y):
            parameter = (border_y - origin_y) / direction_y
            if parameter >= -EPS:
                current_x = origin_x + parameter * direction_x
                if min_x - EPS <= current_x <= max_x + EPS:
                    best_parameter = min(best_parameter, max(parameter, 0.0))

    if best_parameter == float("inf"):
        best_parameter = 2.0 * max(max_x - min_x, max_y - min_y)

    return (
        origin_x + best_parameter * direction_x,
        origin_y + best_parameter * direction_y,
    )


def clip_segment_by_bbox(
    segment_start: Point,
    segment_end: Point,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> Optional[tuple[Point, Point]]:
    start_x, start_y = segment_start
    end_x, end_y = segment_end
    delta_x = end_x - start_x
    delta_y = end_y - start_y

    left_parameter = 0.0
    right_parameter = 1.0

    def apply_clip(p_value: float, q_value: float) -> bool:
        nonlocal left_parameter, right_parameter

        if abs(p_value) < EPS:
            return q_value >= 0.0

        candidate_parameter = q_value / p_value

        if p_value < 0.0:
            if candidate_parameter > right_parameter:
                return False
            if candidate_parameter > left_parameter:
                left_parameter = candidate_parameter
        else:
            if candidate_parameter < left_parameter:
                return False
            if candidate_parameter < right_parameter:
                right_parameter = candidate_parameter

        return left_parameter <= right_parameter

    if not apply_clip(-delta_x, start_x - min_x):
        return None
    if not apply_clip(delta_x, max_x - start_x):
        return None
    if not apply_clip(-delta_y, start_y - min_y):
        return None
    if not apply_clip(delta_y, max_y - start_y):
        return None

    return (
        (start_x + left_parameter * delta_x, start_y + left_parameter * delta_y),
        (start_x + right_parameter * delta_x, start_y + right_parameter * delta_y),
    )


def circumcircle_from_three_points(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> Optional[tuple[float, float, float]]:
    denominator = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(denominator) < EPS * EPS:
        return None

    first_norm = ax * ax + ay * ay
    second_norm = bx * bx + by * by
    third_norm = cx * cx + cy * cy

    center_x = (
        first_norm * (by - cy)
        + second_norm * (cy - ay)
        + third_norm * (ay - by)
    ) / denominator
    center_y = (
        first_norm * (cx - bx)
        + second_norm * (ax - cx)
        + third_norm * (bx - ax)
    ) / denominator
    radius = math.hypot(ax - center_x, ay - center_y)

    return center_x, center_y, radius


def point_is_strictly_inside_circumcircle(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> bool:
    circumcircle = circumcircle_from_three_points(ax, ay, bx, by, cx, cy)
    if circumcircle is None:
        return False

    center_x, center_y, radius = circumcircle
    return math.hypot(px - center_x, py - center_y) < radius - 1e-9


def build_delaunay_triangles_bruteforce(
    active_indices: list[int],
    sites: list[Point],
) -> set[frozenset[int]]:
    if len(active_indices) < 3:
        return set()

    triangles: set[frozenset[int]] = set()

    for first_position in range(len(active_indices)):
        for second_position in range(first_position + 1, len(active_indices)):
            for third_position in range(second_position + 1, len(active_indices)):
                first_index = active_indices[first_position]
                second_index = active_indices[second_position]
                third_index = active_indices[third_position]

                first_point = sites[first_index]
                second_point = sites[second_index]
                third_point = sites[third_index]

                if abs(orient(first_point, second_point, third_point)) < EPS:
                    continue

                has_inner_point = False
                for candidate_index in active_indices:
                    if candidate_index in (first_index, second_index, third_index):
                        continue

                    candidate_point = sites[candidate_index]
                    if point_is_strictly_inside_circumcircle(
                        candidate_point[0],
                        candidate_point[1],
                        first_point[0],
                        first_point[1],
                        second_point[0],
                        second_point[1],
                        third_point[0],
                        third_point[1],
                    ):
                        has_inner_point = True
                        break

                if not has_inner_point:
                    triangles.add(frozenset((first_index, second_index, third_index)))

    return triangles


def convert_delaunay_to_voronoi(
    delaunay_triangles: set[frozenset[int]],
    sites: list[Point],
    active_indices: list[int],
) -> VoronoiDiagramResult:

    triangle_list = [tuple(sorted(triangle)) for triangle in delaunay_triangles]

    circumcenters: dict[tuple[int, int, int], Point] = {}

    for triangle in triangle_list:
        i, j, k = triangle
        ax, ay = sites[i]
        bx, by = sites[j]
        cx, cy = sites[k]

        circle = circumcircle_from_three_points(ax, ay, bx, by, cx, cy)
        if circle is None:
            continue

        circumcenters[triangle] = (circle[0], circle[1])

    edge_to_triangles: dict[tuple[int, int], list[tuple[int, int, int]]] = {}

    for triangle in triangle_list:
        i, j, k = triangle
        for a, b in [(i, j), (j, k), (k, i)]:
            key = tuple(sorted((a, b)))
            edge_to_triangles.setdefault(key, []).append(triangle)

    min_x, max_x, min_y, max_y = build_points_bbox([sites[i] for i in active_indices])

    ridges: list[Ridge] = []

    for (i, j), triangles in edge_to_triangles.items():

        if len(triangles) == 2:
            t1, t2 = triangles

            if t1 not in circumcenters or t2 not in circumcenters:
                continue

            p1 = circumcenters[t1]
            p2 = circumcenters[t2]

            clipped = clip_segment_by_bbox(p1, p2, min_x, max_x, min_y, max_y)
            if clipped:
                ridges.append((i, j, clipped[0], clipped[1]))


        elif len(triangles) == 1:
            t = triangles[0]
            if t not in circumcenters:
                continue

            center = circumcenters[t]
            triangle_vertices = set(t)
            third_index = next(index for index in triangle_vertices if index not in (i, j))
            first_point = sites[i]
            second_point = sites[j]
            third_point = sites[third_index]

            mid_x = 0.5 * (first_point[0] + second_point[0])
            mid_y = 0.5 * (first_point[1] + second_point[1])

            edge_x = second_point[0] - first_point[0]
            edge_y = second_point[1] - first_point[1]
            normal_x = -edge_y
            normal_y = edge_x
            normal_length = math.hypot(normal_x, normal_y)

            if normal_length < EPS:
                continue

            normal_x /= normal_length
            normal_y /= normal_length
            third_vector_x = third_point[0] - mid_x
            third_vector_y = third_point[1] - mid_y

            if normal_x * third_vector_x + normal_y * third_vector_y > 0.0:
                normal_x = -normal_x
                normal_y = -normal_y

            far_point = ray_to_bbox_boundary(
                center,
                (normal_x, normal_y),
                min_x,
                max_x,
                min_y,
                max_y,
            )

            clipped = clip_segment_by_bbox(center, far_point, min_x, max_x, min_y, max_y)

            if clipped is not None:
                ridges.append((i, j, clipped[0], clipped[1]))

    return VoronoiDiagramResult(sites=sites, vertices=list(circumcenters.values()), ridges=ridges)


def build_voronoi_divide_and_conquer(sites: list[Point]) -> VoronoiDiagramResult:
    site_count = len(sites)

    if site_count == 0:
        return VoronoiDiagramResult([], [], [])
    if site_count == 1:
        return VoronoiDiagramResult([sites[0]], [], [])

    sorted_indices = sorted(range(site_count), key=lambda index: (sites[index][0], sites[index][1]))

    def solve_recursively(current_indices: list[int]) -> set[frozenset[int]]:
        if len(current_indices) < 3:
            return set()
        if len(current_indices) == 3:
            return build_delaunay_triangles_bruteforce(current_indices, sites)

        middle_position = len(current_indices) // 2
        solve_recursively(current_indices[:middle_position])
        solve_recursively(current_indices[middle_position:])
        return build_delaunay_triangles_bruteforce(current_indices, sites)

    delaunay_triangles = solve_recursively(sorted_indices)
    return convert_delaunay_to_voronoi(delaunay_triangles, sites, sorted_indices)


def build_delaunay_graph_from_voronoi(voronoi_result: VoronoiDiagramResult) -> set[tuple[int, int]]:
    delaunay_edges: set[tuple[int, int]] = set()

    for first_index, second_index, _, _ in voronoi_result.ridges:
        if first_index == second_index:
            continue
        if first_index < 0 or second_index < 0:
            continue
        delaunay_edges.add((min(first_index, second_index), max(first_index, second_index)))

    return delaunay_edges


def build_delaunay_triangles_from_edges(
    site_count: int,
    delaunay_edges: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    adjacency_list: list[set[int]] = [set() for _ in range(site_count)]

    for first_index, second_index in delaunay_edges:
        adjacency_list[first_index].add(second_index)
        adjacency_list[second_index].add(first_index)

    triangles: set[tuple[int, int, int]] = set()

    for first_index in range(site_count):
        for second_index in adjacency_list[first_index]:
            if second_index <= first_index:
                continue

            common_neighbors = adjacency_list[first_index] & adjacency_list[second_index]
            for third_index in common_neighbors:
                if third_index <= second_index:
                    continue
                triangles.add(tuple(sorted((first_index, second_index, third_index))))

    return sorted(triangles)


def build_voronoi_fortune(sites: list[Point]) -> VoronoiDiagramResult:
    raw_sites, raw_vertices, raw_ridges = build_fortune_voronoi_raw(sites)
    raw_result = VoronoiDiagramResult(
        sites=list(raw_sites),
        vertices=list(raw_vertices),
        ridges=list(raw_ridges),
    )

    triangle_set = build_delaunay_triangles_bruteforce(
        list(range(len(sites))),
        sites
    )

    return convert_delaunay_to_voronoi(
        triangle_set,
        sites,
        list(range(len(sites))),
    )

def closest_pair_from_delaunay_edges(
    sites: list[Point],
    delaunay_edges: set[tuple[int, int]],
) -> tuple[int, int, float]:
    best_first_index = -1
    best_second_index = -1
    best_distance = float("inf")

    for first_index, second_index in delaunay_edges:
        current_distance = math.dist(sites[first_index], sites[second_index])
        if current_distance < best_distance:
            best_first_index = first_index
            best_second_index = second_index
            best_distance = current_distance

    return best_first_index, best_second_index, best_distance


def closest_pair_bruteforce(sites: list[Point]) -> Optional[tuple[int, int, float]]:
    if len(sites) < 2:
        return None

    best_first_index = 0
    best_second_index = 1
    best_distance = math.sqrt(squared_distance(sites[0], sites[1]))

    for first_index in range(len(sites)):
        for second_index in range(first_index + 1, len(sites)):
            current_distance = math.sqrt(squared_distance(sites[first_index], sites[second_index]))
            if current_distance < best_distance:
                best_first_index = first_index
                best_second_index = second_index
                best_distance = current_distance

    return best_first_index, best_second_index, best_distance


def generate_random_sites(
    site_count: int,
    random_generator: np.random.Generator,
    radius: float = 8.0,
) -> list[Point]:
    sites: list[Point] = []

    for _ in range(site_count):
        angle = 2.0 * math.pi * random_generator.random()
        radial_distance = radius * math.sqrt(random_generator.random())
        point_x = radial_distance * math.cos(angle) + random_generator.normal(0.0, 1e-9)
        point_y = radial_distance * math.sin(angle) + random_generator.normal(0.0, 1e-9)
        sites.append((float(point_x), float(point_y)))

    return sites


def plot_results(
    sites: list[Point],
    fortune_result: VoronoiDiagramResult,
    divide_result: VoronoiDiagramResult,
    delaunay_edges: set[tuple[int, int]],
    closest_pair: Optional[tuple[int, int, float]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    min_x, max_x, min_y, max_y = build_points_bbox(sites, padding_ratio=0.08)

    for axis, voronoi_result, title in (
        (axes[0], fortune_result, "Вороной: алгоритм Форчуна + Делоне"),
        (axes[1], divide_result, "Вороной: разделяй и властвуй + Делоне"),
    ):
        axis.set_title(title, fontsize=10)
        axis.set_aspect("equal")
        axis.set_xlim(min_x, max_x)
        axis.set_ylim(min_y, max_y)

        for _, _, ridge_start, ridge_end in voronoi_result.ridges:
            if ridge_start is None or ridge_end is None:
                continue

            axis.plot(
                [ridge_start[0], ridge_end[0]],
                [ridge_start[1], ridge_end[1]],
                "b-",
                linewidth=0.7,
                alpha=0.85,
            )

        for first_index, second_index in delaunay_edges:
            first_point = sites[first_index]
            second_point = sites[second_index]

            axis.plot(
                [first_point[0], second_point[0]],
                [first_point[1], second_point[1]],
                "g--",
                linewidth=0.6,
                alpha=0.7,
            )

        axis.scatter(
            [point[0] for point in sites],
            [point[1] for point in sites],
            c="k",
            s=22,
            zorder=3,
        )

    if closest_pair is not None:
        first_index, second_index, _ = closest_pair
        for axis in axes:
            axis.plot(
                [sites[first_index][0], sites[second_index][0]],
                [sites[first_index][1], sites[second_index][1]],
                "r-",
                linewidth=2.0,
                zorder=4,
            )

    legend_items = [
        Line2D([0], [0], color="b", linewidth=1.2, label="рёбра Вороного"),
        Line2D([0], [0], color="g", linestyle="--", linewidth=1.0, label="рёбра Делоне"),
        Line2D([0], [0], marker="o", color="k", linestyle="None", markersize=6, label="точки P"),
        Line2D([0], [0], color="r", linewidth=2.0, label="ближайшая пара"),
    ]

    figure.legend(handles=legend_items, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()


def main(site_count: int = 32, seed: int = 20, show_plot: bool = True) -> None:

    random_generator = np.random.default_rng(seed)
    sites = generate_random_sites(site_count, random_generator)

    fortune_result = build_voronoi_fortune(sites)
    divide_result = build_voronoi_divide_and_conquer(sites)

    fortune_delaunay_edges = build_delaunay_graph_from_voronoi(fortune_result)
    divide_delaunay_edges = build_delaunay_graph_from_voronoi(divide_result)

    if fortune_delaunay_edges != divide_delaunay_edges:
        print(
            "Предупреждение: графы Делоне, полученные двумя способами, различаются. "
            f"Только у Форчуна: {len(fortune_delaunay_edges - divide_delaunay_edges)}, "
            f"только у D&C: {len(divide_delaunay_edges - fortune_delaunay_edges)}"
        )
    else:
        print("Согласованность: графы Делоне совпадают.")

    delaunay_edges = fortune_delaunay_edges

    closest_pair = closest_pair_from_delaunay_edges(sites, delaunay_edges)
    if closest_pair[0] == -1:
        closest_pair = closest_pair_bruteforce(sites)

    naive_pair = closest_pair_bruteforce(sites)
    assert naive_pair is not None and closest_pair is not None
    assert abs(naive_pair[2] - closest_pair[2]) < 1e-5, (naive_pair, closest_pair)

    delaunay_triangles = build_delaunay_triangles_from_edges(site_count, delaunay_edges)

    print(f"Сайтов: {site_count}")
    print(f"Рёбер Делоне: {len(delaunay_edges)}")
    print(f"Треугольников Делоне: {len(delaunay_triangles)}")
    print(
        f"Ближайшая пара: индексы {closest_pair[0]} и {closest_pair[1]}, "
        f"расстояние {closest_pair[2]:.6f}"
    )

    if show_plot:
        plot_results(sites, fortune_result, divide_result, delaunay_edges, closest_pair)


if __name__ == "__main__":
    main()