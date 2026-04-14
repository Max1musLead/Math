from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay, Voronoi

Point = tuple[float, float]
Edge = tuple[int, int]

@dataclass
class VoronoiDiagramData:
    points: np.ndarray
    voronoi: Voronoi
    finite_segments: list[tuple[Point, Point]]
    rays: list[tuple[Point, Point]]

@dataclass
class DelaunayGraphData:
    points: np.ndarray
    triangles: np.ndarray
    edges: list[Edge]

@dataclass
class ClosestPairData:
    first_index: int
    second_index: int
    distance: float

def squared_distance(first_point: np.ndarray, second_point: np.ndarray) -> float:
    delta_x = first_point[0] - second_point[0]
    delta_y = first_point[1] - second_point[1]
    return delta_x * delta_x + delta_y * delta_y

def distance(first_point: np.ndarray, second_point: np.ndarray) -> float:
    return math.sqrt(squared_distance(first_point, second_point))

def generate_points(point_count: int = 12, seed_value: int = 42) -> np.ndarray:
    generator = np.random.default_rng(seed_value)
    points = generator.uniform(0.0, 10.0, size=(point_count, 2))

    unique_points = []
    used = set()
    for point in points:
        rounded_key = (round(float(point[0]), 8), round(float(point[1]), 8))
        if rounded_key not in used:
            used.add(rounded_key)
            unique_points.append(point)

    while len(unique_points) < point_count:
        new_point = generator.uniform(0.0, 10.0, size=2)
        rounded_key = (round(float(new_point[0]), 8), round(float(new_point[1]), 8))
        if rounded_key not in used:
            used.add(rounded_key)
            unique_points.append(new_point)

    return np.array(unique_points, dtype=float)

def compute_plot_bounds(points: np.ndarray, margin_ratio: float = 0.15) -> tuple[float, float, float, float]:
    minimum_x = float(np.min(points[:, 0]))
    maximum_x = float(np.max(points[:, 0]))
    minimum_y = float(np.min(points[:, 1]))
    maximum_y = float(np.max(points[:, 1]))

    width = maximum_x - minimum_x
    height = maximum_y - minimum_y

    if width < 1e-9:
        width = 1.0
    if height < 1e-9:
        height = 1.0

    margin_x = width * margin_ratio
    margin_y = height * margin_ratio

    return (
        minimum_x - margin_x,
        maximum_x + margin_x,
        minimum_y - margin_y,
        maximum_y + margin_y,
    )

def voronoi_finite_and_infinite_segments(points: np.ndarray, voronoi: Voronoi) -> tuple[list[tuple[Point, Point]], list[tuple[Point, Point]]]:
    finite_segments: list[tuple[Point, Point]] = []
    rays: list[tuple[Point, Point]] = []

    center_point = np.mean(points, axis=0)
    bound_min_x, bound_max_x, bound_min_y, bound_max_y = compute_plot_bounds(points, margin_ratio=0.35)
    bounding_radius = max(bound_max_x - bound_min_x, bound_max_y - bound_min_y) * 2.0

    for ridge_points, ridge_vertices in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        first_site_index, second_site_index = ridge_points
        vertex_a_index, vertex_b_index = ridge_vertices

        if vertex_a_index >= 0 and vertex_b_index >= 0:
            first_vertex = voronoi.vertices[vertex_a_index]
            second_vertex = voronoi.vertices[vertex_b_index]
            finite_segments.append(((float(first_vertex[0]), float(first_vertex[1])), (float(second_vertex[0]), float(second_vertex[1]))))
            continue

        finite_vertex_index = vertex_a_index if vertex_a_index >= 0 else vertex_b_index
        finite_vertex = voronoi.vertices[finite_vertex_index]

        first_site = points[first_site_index]
        second_site = points[second_site_index]

        tangent_vector = second_site - first_site
        tangent_length = np.linalg.norm(tangent_vector)
        if tangent_length < 1e-12:
            continue

        tangent_vector = tangent_vector / tangent_length
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])

        midpoint = (first_site + second_site) / 2.0
        direction_sign = np.sign(np.dot(midpoint - center_point, normal_vector))
        if direction_sign == 0:
            direction_sign = 1.0

        ray_direction = normal_vector * direction_sign
        far_point = finite_vertex + ray_direction * bounding_radius

        rays.append(
            (
                (float(finite_vertex[0]), float(finite_vertex[1])),
                (float(far_point[0]), float(far_point[1])),
            )
        )

    return finite_segments, rays

def build_voronoi_result(points: np.ndarray) -> VoronoiDiagramData:
    voronoi = Voronoi(points)
    finite_segments, rays = voronoi_finite_and_infinite_segments(points, voronoi)
    return VoronoiDiagramData(
        points=points,
        voronoi=voronoi,
        finite_segments=finite_segments,
        rays=rays,
    )

def build_delaunay_graph(points: np.ndarray) -> DelaunayGraphData:
    triangulation = Delaunay(points)
    edge_set: set[Edge] = set()

    for triangle in triangulation.simplices:
        first_index = int(triangle[0])
        second_index = int(triangle[1])
        third_index = int(triangle[2])

        edge_set.add(tuple(sorted((first_index, second_index))))
        edge_set.add(tuple(sorted((second_index, third_index))))
        edge_set.add(tuple(sorted((first_index, third_index))))

    sorted_edges = sorted(edge_set)
    return DelaunayGraphData(
        points=points,
        triangles=triangulation.simplices.copy(),
        edges=sorted_edges,
    )

def find_closest_pair_from_delaunay(points: np.ndarray, delaunay_edges: Iterable[Edge]) -> ClosestPairData:
    best_first_index = -1
    best_second_index = -1
    best_squared_distance = float("inf")

    for first_index, second_index in delaunay_edges:
        current_squared_distance = squared_distance(points[first_index], points[second_index])
        if current_squared_distance < best_squared_distance:
            best_squared_distance = current_squared_distance
            best_first_index = first_index
            best_second_index = second_index

    return ClosestPairData(
        first_index=best_first_index,
        second_index=best_second_index,
        distance=math.sqrt(best_squared_distance),
    )


def find_closest_pair_bruteforce(points: np.ndarray) -> ClosestPairData:
    point_count = len(points)
    best_first_index = -1
    best_second_index = -1
    best_squared_distance = float("inf")

    for first_index in range(point_count):
        for second_index in range(first_index + 1, point_count):
            current_squared_distance = squared_distance(points[first_index], points[second_index])
            if current_squared_distance < best_squared_distance:
                best_squared_distance = current_squared_distance
                best_first_index = first_index
                best_second_index = second_index

    return ClosestPairData(
        first_index=best_first_index,
        second_index=best_second_index,
        distance=math.sqrt(best_squared_distance),
    )


def parabola_x_for_vertical_directrix(site_x: float, site_y: float, sweep_line_x: float, sample_y_values: np.ndarray) -> np.ndarray:
    denominator = 2.0 * (site_x - sweep_line_x)
    result_x_values = np.full_like(sample_y_values, np.nan, dtype=float)

    if abs(denominator) < 1e-12:
        return result_x_values

    numerator = (sample_y_values - site_y) ** 2 + site_x ** 2 - sweep_line_x ** 2
    result_x_values = numerator / denominator
    return result_x_values


def compute_beach_line(points: np.ndarray, sweep_line_x: float, sample_y_values: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    active_points = points[points[:, 0] < sweep_line_x - 1e-9]

    all_parabolas: list[np.ndarray] = []
    if len(active_points) == 0:
        beach_line_x_values = np.full_like(sample_y_values, np.nan, dtype=float)
        ownership_indices = np.full_like(sample_y_values, -1, dtype=int)
        return beach_line_x_values, all_parabolas, ownership_indices

    for site_x, site_y in active_points:
        parabola_x_values = parabola_x_for_vertical_directrix(site_x, site_y, sweep_line_x, sample_y_values)
        all_parabolas.append(parabola_x_values)

    parabola_matrix = np.vstack(all_parabolas)

    valid_mask = np.isfinite(parabola_matrix)
    masked_matrix = np.where(valid_mask, parabola_matrix, -np.inf)

    ownership_indices = np.argmax(masked_matrix, axis=0)
    beach_line_x_values = masked_matrix[ownership_indices, np.arange(len(sample_y_values))]
    beach_line_x_values[np.isneginf(beach_line_x_values)] = np.nan

    return beach_line_x_values, all_parabolas, ownership_indices


def draw_fortune_sweep(ax: plt.Axes, points: np.ndarray, sweep_line_x: float) -> None:
    minimum_x, maximum_x, minimum_y, maximum_y = compute_plot_bounds(points, margin_ratio=0.25)

    sample_y_values = np.linspace(minimum_y, maximum_y, 600)
    beach_line_x_values, all_parabolas, ownership_indices = compute_beach_line(points, sweep_line_x, sample_y_values)

    ax.scatter(points[:, 0], points[:, 1], s=45, color="black", zorder=4, label="Точки множества P")

    for parabola_x_values in all_parabolas:
        valid_mask = np.isfinite(parabola_x_values)
        ax.plot(parabola_x_values[valid_mask], sample_y_values[valid_mask], linestyle="--", linewidth=1.0, alpha=0.6)

    valid_beach_mask = np.isfinite(beach_line_x_values)
    ax.plot(
        beach_line_x_values[valid_beach_mask],
        sample_y_values[valid_beach_mask],
        linewidth=2.5,
        color="red",
        label="Береговая линия",
        zorder=5,
    )

    ax.axvline(
        x=sweep_line_x,
        linestyle="-.",
        linewidth=2.0,
        color="blue",
        label="Заметающая прямая",
        zorder=3,
    )

    for point_index, point in enumerate(points):
        ax.text(point[0] + 0.06, point[1] + 0.06, f"P{point_index + 1}", fontsize=9)

    ax.set_xlim(minimum_x, maximum_x)
    ax.set_ylim(minimum_y, maximum_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Алгоритм Форчуна: заметающая прямая и береговая линия")
    ax.grid(True, alpha=0.25)
    ax.legend()


def bisection_root(function, left_value: float, right_value: float, iteration_count: int = 60) -> float | None:
    function_left = function(left_value)
    function_right = function(right_value)

    if np.isnan(function_left) or np.isnan(function_right):
        return None

    if abs(function_left) < 1e-12:
        return left_value
    if abs(function_right) < 1e-12:
        return right_value

    if function_left * function_right > 0:
        return None

    current_left = left_value
    current_right = right_value

    for _ in range(iteration_count):
        middle_value = (current_left + current_right) / 2.0
        function_middle = function(middle_value)

        if np.isnan(function_middle):
            return None

        if abs(function_middle) < 1e-12:
            return middle_value

        if function_left * function_middle <= 0:
            current_right = middle_value
            function_right = function_middle
        else:
            current_left = middle_value
            function_left = function_middle

    return (current_left + current_right) / 2.0


def compute_separator_polyline(left_points: np.ndarray, right_points: np.ndarray, bounds: tuple[float, float, float, float], sample_count: int = 400) -> tuple[np.ndarray, list[tuple[int, int]]]:
    minimum_x, maximum_x, minimum_y, maximum_y = bounds
    y_values = np.linspace(maximum_y, minimum_y, sample_count)

    separator_points: list[list[float]] = []
    active_pairs: list[tuple[int, int]] = []

    def left_minus_right_distance(current_x: float, current_y: float) -> float:
        test_point = np.array([current_x, current_y], dtype=float)

        left_distances = np.sum((left_points - test_point) ** 2, axis=1)
        right_distances = np.sum((right_points - test_point) ** 2, axis=1)

        minimum_left_distance = float(np.min(left_distances))
        minimum_right_distance = float(np.min(right_distances))

        return minimum_left_distance - minimum_right_distance

    for current_y in y_values:
        function = lambda current_x: left_minus_right_distance(current_x, current_y)
        root_x = bisection_root(function, minimum_x, maximum_x)

        if root_x is None:
            continue

        test_point = np.array([root_x, current_y], dtype=float)
        left_distances = np.sum((left_points - test_point) ** 2, axis=1)
        right_distances = np.sum((right_points - test_point) ** 2, axis=1)

        nearest_left_index = int(np.argmin(left_distances))
        nearest_right_index = int(np.argmin(right_distances))

        separator_points.append([root_x, current_y])
        active_pairs.append((nearest_left_index, nearest_right_index))

    if len(separator_points) == 0:
        return np.empty((0, 2), dtype=float), []

    return np.array(separator_points, dtype=float), active_pairs


def recursive_partition(points: np.ndarray) -> list[np.ndarray]:
    if len(points) <= 3:
        return [points]

    sorted_indices = np.argsort(points[:, 0], kind="mergesort")
    sorted_points = points[sorted_indices]

    middle_index = len(sorted_points) // 2
    left_part = sorted_points[:middle_index]
    right_part = sorted_points[middle_index:]

    result = []
    result.extend(recursive_partition(left_part))
    result.extend(recursive_partition(right_part))
    return result


def draw_divide_and_conquer(ax: plt.Axes, points: np.ndarray) -> None:
    sorted_indices = np.argsort(points[:, 0], kind="mergesort")
    sorted_points = points[sorted_indices]

    middle_index = len(sorted_points) // 2
    left_points = sorted_points[:middle_index]
    right_points = sorted_points[middle_index:]

    bounds = compute_plot_bounds(points, margin_ratio=0.25)
    separator_polyline, active_pairs = compute_separator_polyline(left_points, right_points, bounds, sample_count=500)

    ax.scatter(left_points[:, 0], left_points[:, 1], color="green", s=45, label="Левая половина L", zorder=4)
    ax.scatter(right_points[:, 0], right_points[:, 1], color="orange", s=45, label="Правая половина R", zorder=4)

    split_x_value = (left_points[-1, 0] + right_points[0, 0]) / 2.0
    ax.axvline(split_x_value, color="gray", linestyle="--", linewidth=1.5, label="Разделение по x")

    if len(separator_polyline) > 1:
        ax.plot(
            separator_polyline[:, 0],
            separator_polyline[:, 1],
            color="red",
            linewidth=2.5,
            label="Разделяющая ломаная",
            zorder=5,
        )

    step_for_labels = max(1, len(separator_polyline) // 8)
    for separator_index in range(0, len(separator_polyline), step_for_labels):
        point_x = separator_polyline[separator_index, 0]
        point_y = separator_polyline[separator_index, 1]
        left_local_index, right_local_index = active_pairs[separator_index]
        left_global_index = np.where((points == left_points[left_local_index]).all(axis=1))[0][0]
        right_global_index = np.where((points == right_points[right_local_index]).all(axis=1))[0][0]
        ax.text(
            point_x + 0.05,
            point_y,
            f"P{left_global_index + 1}-P{right_global_index + 1}",
            fontsize=8,
            color="darkred",
        )

    for point_index, point in enumerate(points):
        ax.text(point[0] + 0.06, point[1] + 0.06, f"P{point_index + 1}", fontsize=9)

    minimum_x, maximum_x, minimum_y, maximum_y = bounds
    ax.set_xlim(minimum_x, maximum_x)
    ax.set_ylim(minimum_y, maximum_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Разделяй и властвуй: разделяющая ломаная между L и R")
    ax.grid(True, alpha=0.25)
    ax.legend()


def draw_voronoi(ax: plt.Axes, voronoi_data: VoronoiDiagramData, closest_pair: ClosestPairData | None = None) -> None:
    points = voronoi_data.points

    for first_segment_point, second_segment_point in voronoi_data.finite_segments:
        ax.plot(
            [first_segment_point[0], second_segment_point[0]],
            [first_segment_point[1], second_segment_point[1]],
            color="tab:blue",
            linewidth=1.5,
        )

    for first_ray_point, second_ray_point in voronoi_data.rays:
        ax.plot(
            [first_ray_point[0], second_ray_point[0]],
            [first_ray_point[1], second_ray_point[1]],
            color="tab:blue",
            linewidth=1.2,
            linestyle="--",
        )

    ax.scatter(points[:, 0], points[:, 1], color="black", s=45, zorder=5)

    if closest_pair is not None:
        first_point = points[closest_pair.first_index]
        second_point = points[closest_pair.second_index]
        ax.plot(
            [first_point[0], second_point[0]],
            [first_point[1], second_point[1]],
            color="red",
            linewidth=2.5,
            label=f"Ближайшая пара: P{closest_pair.first_index + 1}-P{closest_pair.second_index + 1}",
            zorder=6,
        )

    for point_index, point in enumerate(points):
        ax.text(point[0] + 0.06, point[1] + 0.06, f"P{point_index + 1}", fontsize=9)

    minimum_x, maximum_x, minimum_y, maximum_y = compute_plot_bounds(points, margin_ratio=0.25)
    ax.set_xlim(minimum_x, maximum_x)
    ax.set_ylim(minimum_y, maximum_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Диаграмма Вороного")
    ax.grid(True, alpha=0.25)
    if closest_pair is not None:
        ax.legend()


def draw_delaunay(ax: plt.Axes, delaunay_data: DelaunayGraphData, closest_pair: ClosestPairData | None = None) -> None:
    points = delaunay_data.points

    for first_index, second_index in delaunay_data.edges:
        first_point = points[first_index]
        second_point = points[second_index]
        ax.plot(
            [first_point[0], second_point[0]],
            [first_point[1], second_point[1]],
            color="tab:green",
            linewidth=1.3,
        )

    ax.scatter(points[:, 0], points[:, 1], color="black", s=45, zorder=5)

    if closest_pair is not None:
        first_point = points[closest_pair.first_index]
        second_point = points[closest_pair.second_index]
        ax.plot(
            [first_point[0], second_point[0]],
            [first_point[1], second_point[1]],
            color="red",
            linewidth=2.5,
            label=f"Ближайшая пара: P{closest_pair.first_index + 1}-P{closest_pair.second_index + 1}",
            zorder=6,
        )

    for point_index, point in enumerate(points):
        ax.text(point[0] + 0.06, point[1] + 0.06, f"P{point_index + 1}", fontsize=9)

    minimum_x, maximum_x, minimum_y, maximum_y = compute_plot_bounds(points, margin_ratio=0.25)
    ax.set_xlim(minimum_x, maximum_x)
    ax.set_ylim(minimum_y, maximum_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Триангуляция Делоне — граф, двойственный диаграмме Вороного")
    ax.grid(True, alpha=0.25)
    if closest_pair is not None:
        ax.legend()


def print_results(points: np.ndarray, closest_pair_delaunay: ClosestPairData, closest_pair_bruteforce: ClosestPairData) -> None:
    first_delaunay_point = points[closest_pair_delaunay.first_index]
    second_delaunay_point = points[closest_pair_delaunay.second_index]

    print("Точки множества P:")
    for point_index, point in enumerate(points):
        print(f"P{point_index + 1} = ({point[0]:.4f}, {point[1]:.4f})")

    print()
    print("Ближайшая пара по ребрам Делоне:")
    print(
        f"P{closest_pair_delaunay.first_index + 1} и P{closest_pair_delaunay.second_index + 1}, "
        f"расстояние = {closest_pair_delaunay.distance:.6f}"
    )
    print(
        f"Координаты: ({first_delaunay_point[0]:.4f}, {first_delaunay_point[1]:.4f}) и "
        f"({second_delaunay_point[0]:.4f}, {second_delaunay_point[1]:.4f})"
    )

    print()
    print("Проверка полным перебором:")
    print(
        f"P{closest_pair_bruteforce.first_index + 1} и P{closest_pair_bruteforce.second_index + 1}, "
        f"расстояние = {closest_pair_bruteforce.distance:.6f}"
    )

def draw_divide_and_conquer_result(
        ax: plt.Axes,
        voronoi_data: VoronoiDiagramData,
        split_x_value: float,
        closest_pair: ClosestPairData | None = None,
) -> None:
    points = voronoi_data.points

    for first_segment_point, second_segment_point in voronoi_data.finite_segments:
        ax.plot(
            [first_segment_point[0], second_segment_point[0]],
            [first_segment_point[1], second_segment_point[1]],
            color="tab:blue",
            linewidth=1.5,
        )

    for first_ray_point, second_ray_point in voronoi_data.rays:
        ax.plot(
            [first_ray_point[0], second_ray_point[0]],
            [first_ray_point[1], second_ray_point[1]],
            color="tab:blue",
            linewidth=1.2,
            linestyle="--",
        )

    ax.scatter(points[:, 0], points[:, 1], color="black", s=45, zorder=5)

    ax.axvline(
        x=split_x_value,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Граница разбиения L / R",
        zorder=3,
    )

    if closest_pair is not None:
        first_point = points[closest_pair.first_index]
        second_point = points[closest_pair.second_index]
        ax.plot(
            [first_point[0], second_point[0]],
            [first_point[1], second_point[1]],
            color="red",
            linewidth=2.5,
            label=f"Ближайшая пара: P{closest_pair.first_index + 1}-P{closest_pair.second_index + 1}",
            zorder=6,
        )

    for point_index, point in enumerate(points):
        ax.text(point[0] + 0.06, point[1] + 0.06, f"P{point_index + 1}", fontsize=9)

    minimum_x, maximum_x, minimum_y, maximum_y = compute_plot_bounds(points, margin_ratio=0.25)
    ax.set_xlim(minimum_x, maximum_x)
    ax.set_ylim(minimum_y, maximum_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Результат после объединения (divide-and-conquer)")
    ax.grid(True, alpha=0.25)
    ax.legend()

def build_animated_voronoi_segments(
    points: np.ndarray,
    voronoi: Voronoi,
) -> list[dict]:
    animated_segments: list[dict] = []

    center_point = np.mean(points, axis=0)
    bound_min_x, bound_max_x, bound_min_y, bound_max_y = compute_plot_bounds(points, margin_ratio=0.35)
    bounding_radius = max(bound_max_x - bound_min_x, bound_max_y - bound_min_y) * 2.0

    for ridge_points, ridge_vertices in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        first_site_index, second_site_index = ridge_points
        vertex_a_index, vertex_b_index = ridge_vertices

        activation_x = max(
            float(points[first_site_index][0]),
            float(points[second_site_index][0]),
        )

        if vertex_a_index >= 0 and vertex_b_index >= 0:
            first_vertex = voronoi.vertices[vertex_a_index]
            second_vertex = voronoi.vertices[vertex_b_index]

            animated_segments.append(
                {
                    "kind": "segment",
                    "activation_x": activation_x,
                    "start_point": np.array([float(first_vertex[0]), float(first_vertex[1])], dtype=float),
                    "end_point": np.array([float(second_vertex[0]), float(second_vertex[1])], dtype=float),
                }
            )
            continue

        finite_vertex_index = vertex_a_index if vertex_a_index >= 0 else vertex_b_index
        finite_vertex = voronoi.vertices[finite_vertex_index]

        first_site = points[first_site_index]
        second_site = points[second_site_index]

        tangent_vector = second_site - first_site
        tangent_length = np.linalg.norm(tangent_vector)
        if tangent_length < 1e-12:
            continue

        tangent_vector = tangent_vector / tangent_length
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]], dtype=float)

        midpoint = (first_site + second_site) / 2.0
        direction_sign = np.sign(np.dot(midpoint - center_point, normal_vector))
        if direction_sign == 0:
            direction_sign = 1.0

        ray_direction = normal_vector * direction_sign
        far_point = finite_vertex + ray_direction * bounding_radius

        animated_segments.append(
            {
                "kind": "ray",
                "activation_x": activation_x,
                "start_point": np.array([float(finite_vertex[0]), float(finite_vertex[1])], dtype=float),
                "end_point": np.array([float(far_point[0]), float(far_point[1])], dtype=float),
            }
        )

    return animated_segments

def draw_progressive_voronoi_for_sweep(
    ax: plt.Axes,
    animated_segments: list[dict],
    sweep_line_x: float,
    color: str = "tab:blue",
) -> None:
    first_label_used = False

    for animated_segment in animated_segments:
        if sweep_line_x < animated_segment["activation_x"]:
            continue

        start_point = animated_segment["start_point"]
        end_point = animated_segment["end_point"]
        line_style = "--" if animated_segment["kind"] == "ray" else "-"

        current_label = None
        if not first_label_used:
            current_label = "Рёбра Вороного"
            first_label_used = True

        ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            color=color,
            linewidth=1.6 if animated_segment["kind"] == "segment" else 1.3,
            linestyle=line_style,
            zorder=2,
            label=current_label,
        )

def save_fortune_sweep_gif(
    points: np.ndarray,
    output_path: str = "fortune_sweep.gif",
    frame_count: int = 50,
    frames_per_second: int = 8,
) -> None:
    minimum_x, maximum_x, minimum_y, maximum_y = compute_plot_bounds(points, margin_ratio=0.25)

    voronoi_data = build_voronoi_result(points)
    animated_segments = build_animated_voronoi_segments(points, voronoi_data.voronoi)

    sweep_positions = np.linspace(minimum_x, maximum_x, frame_count)

    figure, ax = plt.subplots(figsize=(7, 7))

    def update(frame_index: int):
        ax.clear()

        sweep_line_x = float(sweep_positions[frame_index])
        sample_y_values = np.linspace(minimum_y, maximum_y, 600)
        beach_line_x_values, all_parabolas, _ = compute_beach_line(points, sweep_line_x, sample_y_values)

        draw_progressive_voronoi_for_sweep(
            ax=ax,
            animated_segments=animated_segments,
            sweep_line_x=sweep_line_x,
            color="tab:blue",
        )

        ax.scatter(points[:, 0], points[:, 1], s=45, color="black", zorder=4, label="Точки множества P")

        for parabola_x_values in all_parabolas:
            valid_mask = np.isfinite(parabola_x_values)
            ax.plot(
                parabola_x_values[valid_mask],
                sample_y_values[valid_mask],
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                color="gray",
            )

        valid_beach_mask = np.isfinite(beach_line_x_values)
        ax.plot(
            beach_line_x_values[valid_beach_mask],
            sample_y_values[valid_beach_mask],
            linewidth=2.5,
            color="red",
            label="Береговая линия",
            zorder=5,
        )

        ax.axvline(
            x=sweep_line_x,
            linestyle="-.",
            linewidth=2.0,
            color="blue",
            label="Заметающая прямая",
            zorder=3,
        )

        for point_index, point in enumerate(points):
            ax.text(point[0] + 0.06, point[1] + 0.06, f"P{point_index + 1}", fontsize=9)

        ax.set_xlim(minimum_x, maximum_x)
        ax.set_ylim(minimum_y, maximum_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Алгоритм Форчуна: заметающая прямая и построение диаграммы Вороного")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 // frames_per_second,
        repeat=True,
    )

    animation.save(output_path, writer=PillowWriter(fps=frames_per_second))
    plt.close(figure)
    print(f"GIF для заметающей прямой сохранён: {output_path}")

def main() -> None:
    points = generate_points(point_count=32, seed_value=7)

    voronoi_data = build_voronoi_result(points)
    delaunay_data = build_delaunay_graph(points)

    closest_pair_delaunay = find_closest_pair_from_delaunay(points, delaunay_data.edges)
    closest_pair_bruteforce = find_closest_pair_bruteforce(points)

    print_results(points, closest_pair_delaunay, closest_pair_bruteforce)

    minimum_x, maximum_x, minimum_y, maximum_y = compute_plot_bounds(points, margin_ratio=0.25)
    sweep_line_x = minimum_x + 0.72 * (maximum_x - minimum_x)

    sorted_indices = np.argsort(points[:, 0], kind="mergesort")
    sorted_points = points[sorted_indices]
    middle_index = len(sorted_points) // 2
    left_points = sorted_points[:middle_index]
    right_points = sorted_points[middle_index:]
    split_x_value = (left_points[-1, 0] + right_points[0, 0]) / 2.0

    figure, axes = plt.subplots(2, 3, figsize=(20, 12))

    draw_fortune_sweep(axes[0, 0], points, sweep_line_x)
    draw_divide_and_conquer(axes[0, 1], points)
    draw_divide_and_conquer_result(
        axes[0, 2],
        voronoi_data,
        split_x_value,
        closest_pair_delaunay,
    )

    draw_voronoi(axes[1, 0], voronoi_data, closest_pair_delaunay)
    draw_delaunay(axes[1, 1], delaunay_data, closest_pair_delaunay)

    axes[1, 2].axis("off")

    figure.suptitle("", fontsize=14)
    plt.tight_layout()
    plt.show()

    # save_fortune_sweep_gif(
    #     points=points,
    #     output_path="fortune_sweep.gif",
    #     frame_count=50,
    #     frames_per_second=8,
    # )

if __name__ == "__main__":
    main()