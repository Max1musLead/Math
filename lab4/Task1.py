from __future__ import annotations
import math
from typing import Iterable
import numpy as np
from matplotlib import pyplot as plt

EPSILON = 1e-9
Point = tuple[float, float]

def orient(point_a: Point, point_b: Point, point_c: Point) -> float:
    return (
        (point_b[0] - point_a[0]) * (point_c[1] - point_a[1])
        - (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])
    )

def distance_squared(point_a: Point, point_b: Point) -> float:
    delta_x = point_a[0] - point_b[0]
    delta_y = point_a[1] - point_b[1]
    return delta_x * delta_x + delta_y * delta_y

def remove_duplicate_points(points: Iterable[Point]) -> list[Point]:
    unique_points: list[Point] = []
    used_points: set[tuple[float, float]] = set()

    for point in points:
        rounded_point = (round(point[0], 12), round(point[1], 12))
        if rounded_point not in used_points:
            used_points.add(rounded_point)
            unique_points.append(point)

    return unique_points

def generate_points_in_disk(
    point_count: int,
    center: Point,
    radius: float,
    random_generator: np.random.Generator,
) -> list[Point]:
    points: list[Point] = []

    for _ in range(point_count):
        angle = 2.0 * math.pi * random_generator.random()
        random_radius = radius * math.sqrt(random_generator.random())
        x_value = center[0] + random_radius * math.cos(angle)
        y_value = center[1] + random_radius * math.sin(angle)
        points.append((x_value, y_value))

    return points


def generate_point_sets(
    g_count: int,
    d_count: int,
    random_generator: np.random.Generator,
) -> tuple[list[Point], list[Point]]:
    g_points = generate_points_in_disk(g_count, center=(0.0, 0.0), radius=6.0, random_generator=random_generator)
    d_points = generate_points_in_disk(d_count, center=(4.0, 1.0), radius=5.5, random_generator=random_generator)
    return g_points, d_points

def graham_hull(points: list[Point]) -> list[Point]:
    unique_points = remove_duplicate_points(points)

    if len(unique_points) <= 2:
        return unique_points

    start_point = min(unique_points, key=lambda point: (point[1], point[0]))
    other_points = [point for point in unique_points if point != start_point]

    other_points.sort(
        key=lambda point: (
            math.atan2(point[1] - start_point[1], point[0] - start_point[0]),
            distance_squared(start_point, point),
        )
    )

    filtered_points: list[Point] = []
    for point in other_points:
        while filtered_points:
            previous_angle = math.atan2(
                filtered_points[-1][1] - start_point[1],
                filtered_points[-1][0] - start_point[0],
            )
            current_angle = math.atan2(
                point[1] - start_point[1],
                point[0] - start_point[0],
            )
            if abs(previous_angle - current_angle) > EPSILON:
                break
            if distance_squared(start_point, point) >= distance_squared(start_point, filtered_points[-1]):
                filtered_points.pop()
            else:
                break
        else:
            filtered_points.append(point)
            continue

        if not filtered_points or abs(
            math.atan2(filtered_points[-1][1] - start_point[1], filtered_points[-1][0] - start_point[0]) - current_angle
        ) > EPSILON:
            filtered_points.append(point)

    if len(filtered_points) == 0:
        return [start_point]
    if len(filtered_points) == 1:
        return [start_point, filtered_points[0]]

    stack: list[Point] = [start_point, filtered_points[0]]

    for point in filtered_points[1:]:
        while len(stack) >= 2 and orient(stack[-2], stack[-1], point) <= EPSILON:
            stack.pop()
        stack.append(point)

    return stack

def jarvis_hull(points: list[Point]) -> list[Point]:
    unique_points = remove_duplicate_points(points)

    if len(unique_points) <= 2:
        return unique_points

    start_point_index = min(
        range(len(unique_points)),
        key=lambda index: (unique_points[index][1], unique_points[index][0]),
    )

    hull: list[Point] = []
    current_index = start_point_index

    while True:
        hull.append(unique_points[current_index])
        next_index = None

        for point_index in range(len(unique_points)):
            if point_index == current_index:
                continue

            if next_index is None:
                next_index = point_index
                continue

            orientation_value = orient(
                unique_points[current_index],
                unique_points[next_index],
                unique_points[point_index],
            )

            if orientation_value < -EPSILON:
                next_index = point_index
            elif abs(orientation_value) <= EPSILON:
                if distance_squared(unique_points[current_index], unique_points[point_index]) > distance_squared(
                    unique_points[current_index],
                    unique_points[next_index],
                ):
                    next_index = point_index

        current_index = next_index

        if current_index == start_point_index:
            break

    return hull

def polygon_signed_area(polygon: list[Point]) -> float:
    if len(polygon) < 3:
        return 0.0

    area_value = 0.0
    point_count = len(polygon)

    for point_index in range(point_count):
        x1_value, y1_value = polygon[point_index]
        x2_value, y2_value = polygon[(point_index + 1) % point_count]
        area_value += x1_value * y2_value - x2_value * y1_value

    return 0.5 * area_value

def polygon_area(polygon: list[Point]) -> float:
    return abs(polygon_signed_area(polygon))

def polygon_perimeter(polygon: list[Point]) -> float:
    if len(polygon) < 2:
        return 0.0

    perimeter_value = 0.0
    point_count = len(polygon)

    for point_index in range(point_count):
        point_a = polygon[point_index]
        point_b = polygon[(point_index + 1) % point_count]
        perimeter_value += math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])

    return perimeter_value

def ensure_counterclockwise(polygon: list[Point]) -> list[Point]:
    if polygon_signed_area(polygon) < 0.0:
        return polygon[::-1]
    return polygon

def point_on_segment(point: Point, segment_start: Point, segment_end: Point) -> bool:
    if abs(orient(segment_start, segment_end, point)) > EPSILON:
        return False

    return (
        min(segment_start[0], segment_end[0]) - EPSILON <= point[0] <= max(segment_start[0], segment_end[0]) + EPSILON
        and min(segment_start[1], segment_end[1]) - EPSILON <= point[1] <= max(segment_start[1], segment_end[1]) + EPSILON
    )

def point_in_polygon(point: Point, polygon: list[Point]) -> str:
    if len(polygon) < 3:
        return "OUTSIDE"

    winding_number = 0
    point_x, point_y = point

    for point_index in range(len(polygon)):
        point_a = polygon[point_index]
        point_b = polygon[(point_index + 1) % len(polygon)]

        if point_on_segment(point, point_a, point_b):
            return "BOUNDARY"

        if point_a[1] <= point_y:
            if point_b[1] > point_y and orient(point_a, point_b, point) > EPSILON:
                winding_number += 1
        else:
            if point_b[1] <= point_y and orient(point_a, point_b, point) < -EPSILON:
                winding_number -= 1

    if winding_number != 0:
        return "INSIDE"
    return "OUTSIDE"

def strictly_inside_polygon(point: Point, polygon: list[Point]) -> bool:
    return point_in_polygon(point, polygon) == "INSIDE"

def cross_product(vector_a: tuple[float, float], vector_b: tuple[float, float]) -> float:
    return vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]

def point_inside_halfplane(point: Point, line_start: Point, line_end: Point) -> bool:
    return orient(line_start, line_end, point) >= -EPSILON

def segment_line_intersection(
    segment_start: Point,
    segment_end: Point,
    line_start: Point,
    line_end: Point,
) -> Point | None:
    line_direction = (line_end[0] - line_start[0], line_end[1] - line_start[1])
    segment_direction = (segment_end[0] - segment_start[0], segment_end[1] - segment_start[1])

    denominator = cross_product(line_direction, segment_direction)

    if abs(denominator) < EPSILON:
        return None

    start_difference = (segment_start[0] - line_start[0], segment_start[1] - line_start[1])
    parameter_u = -cross_product(line_direction, start_difference) / denominator

    if -EPSILON <= parameter_u <= 1.0 + EPSILON:
        return (
            segment_start[0] + parameter_u * segment_direction[0],
            segment_start[1] + parameter_u * segment_direction[1],
        )

    return None

def convex_polygon_intersection(first_polygon: list[Point], second_polygon: list[Point]) -> list[Point]:
    if len(first_polygon) < 3 or len(second_polygon) < 3:
        return []

    subject_polygon = ensure_counterclockwise(first_polygon)
    clipping_polygon = ensure_counterclockwise(second_polygon)

    def clip_polygon(polygon: list[Point], clip_start: Point, clip_end: Point) -> list[Point]:
        if not polygon:
            return []

        result_polygon: list[Point] = []
        previous_point = polygon[-1]
        previous_inside = point_inside_halfplane(previous_point, clip_start, clip_end)

        for current_point in polygon:
            current_inside = point_inside_halfplane(current_point, clip_start, clip_end)

            if current_inside:
                if not previous_inside:
                    intersection_point = segment_line_intersection(previous_point, current_point, clip_start, clip_end)
                    if intersection_point is not None:
                        result_polygon.append(intersection_point)
                result_polygon.append(current_point)
            elif previous_inside:
                intersection_point = segment_line_intersection(previous_point, current_point, clip_start, clip_end)
                if intersection_point is not None:
                    result_polygon.append(intersection_point)

            previous_point = current_point
            previous_inside = current_inside

        return result_polygon

    result_polygon = subject_polygon

    for point_index in range(len(clipping_polygon)):
        clip_start = clipping_polygon[point_index]
        clip_end = clipping_polygon[(point_index + 1) % len(clipping_polygon)]
        result_polygon = clip_polygon(result_polygon, clip_start, clip_end)

        if not result_polygon:
            return []

    cleaned_polygon: list[Point] = []
    for point in result_polygon:
        if cleaned_polygon and distance_squared(point, cleaned_polygon[-1]) < EPSILON * EPSILON:
            continue
        cleaned_polygon.append(point)

    if len(cleaned_polygon) >= 2 and distance_squared(cleaned_polygon[0], cleaned_polygon[-1]) < EPSILON * EPSILON:
        cleaned_polygon.pop()

    return cleaned_polygon

def plot_result(
    g_points: list[Point],
    d_points: list[Point],
    g_hull: list[Point],
    d_hull: list[Point],
    intersection_polygon: list[Point],
    inner_g_points: list[Point],
    inner_d_points: list[Point],
) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))

    def closed_ring(polygon: list[Point]) -> tuple[list[float], list[float]]:
        x_values = [point[0] for point in polygon] + [polygon[0][0]]
        y_values = [point[1] for point in polygon] + [polygon[0][1]]
        return x_values, y_values

    g_x_values, g_y_values = zip(*g_points)
    d_x_values, d_y_values = zip(*d_points)

    axis.scatter(g_x_values, g_y_values, s=30, label="Точки G")
    axis.scatter(d_x_values, d_y_values, s=30, label="Точки D")

    if len(g_hull) >= 2:
        hull_x_values, hull_y_values = closed_ring(g_hull)
        axis.plot(hull_x_values, hull_y_values, linewidth=2.0, label="Оболочка G (Грэхем)")

    if len(d_hull) >= 2:
        hull_x_values, hull_y_values = closed_ring(d_hull)
        axis.plot(hull_x_values, hull_y_values, linestyle="--", linewidth=2.0, label="Оболочка D (Джарвис)")

    if len(intersection_polygon) >= 3:
        intersection_x_values, intersection_y_values = closed_ring(intersection_polygon)
        axis.plot(intersection_x_values, intersection_y_values, linestyle="-.", linewidth=2.0, label="Пересечение P")

    if inner_g_points:
        inner_g_x_values, inner_g_y_values = zip(*inner_g_points)
        axis.scatter(inner_g_x_values, inner_g_y_values, s=80, marker="*", label="Точки G внутри P")

    if inner_d_points:
        inner_d_x_values, inner_d_y_values = zip(*inner_d_points)
        axis.scatter(inner_d_x_values, inner_d_y_values, s=80, marker="P", label="Точки D внутри P")

    axis.set_aspect("equal", adjustable="box")
    axis.grid(True)
    axis.legend()
    plt.tight_layout()
    plt.show()

def main() -> None:
    random_generator = np.random.default_rng(15)

    g_count = 26
    d_count = 25

    g_points, d_points = generate_point_sets(g_count, d_count, random_generator)

    g_hull = ensure_counterclockwise(graham_hull(g_points))
    d_hull = ensure_counterclockwise(jarvis_hull(d_points))

    g_perimeter = polygon_perimeter(g_hull)
    g_area = polygon_area(g_hull)

    d_perimeter = polygon_perimeter(d_hull)
    d_area = polygon_area(d_hull)

    print(f"Оболочка G, алгоритм Грэхема:")
    print(f"Периметр = {g_perimeter:.6f}")
    print(f"Площадь = {g_area:.6f}")
    print()

    print(f"Оболочка D, алгоритм Джарвиса:")
    print(f"Периметр = {d_perimeter:.6f}")
    print(f"Площадь = {d_area:.6f}")
    print()

    intersection_polygon = convex_polygon_intersection(g_hull, d_hull)

    if len(intersection_polygon) >= 3:
        intersection_perimeter = polygon_perimeter(intersection_polygon)
        intersection_area = polygon_area(intersection_polygon)

        print("Пересечение P = G ∩ D:")
        print(f"Периметр = {intersection_perimeter:.6f}")
        print(f"Площадь = {intersection_area:.6f}")
        print()
    else:
        print("Пересечение P пустое или вырожденное.")
        print()

    inner_g_points = [point for point in g_points if strictly_inside_polygon(point, intersection_polygon)]
    inner_d_points = [point for point in d_points if strictly_inside_polygon(point, intersection_polygon)]

    print(f"Точки множества G строго внутри P: {len(inner_g_points)}")
    print(inner_g_points)
    print()

    print(f"Точки множества D строго внутри P: {len(inner_d_points)}")
    print(inner_d_points)
    print()

    plot_result(
        g_points,
        d_points,
        g_hull,
        d_hull,
        intersection_polygon,
        inner_g_points,
        inner_d_points,
    )


if __name__ == "__main__":
    main()