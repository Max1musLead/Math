import math
import matplotlib.pyplot as plt

epsilon_value = 1e-9

def cross_product(point_a, point_b, point_c):
    return (
        (point_b[0] - point_a[0]) * (point_c[1] - point_a[1])
        - (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])
    )


def distance_squared(point_a, point_b):
    delta_x = point_a[0] - point_b[0]
    delta_y = point_a[1] - point_b[1]
    return delta_x * delta_x + delta_y * delta_y


def polygon_area(polygon_points):
    if len(polygon_points) < 3:
        return 0.0

    doubled_area = 0.0
    points_count = len(polygon_points)

    for point_index in range(points_count):
        next_index = (point_index + 1) % points_count
        doubled_area += (
            polygon_points[point_index][0] * polygon_points[next_index][1]
            - polygon_points[next_index][0] * polygon_points[point_index][1]
        )

    return abs(doubled_area) / 2.0


def polygon_perimeter(polygon_points):
    if len(polygon_points) < 2:
        return 0.0

    perimeter_value = 0.0
    points_count = len(polygon_points)

    for point_index in range(points_count):
        next_index = (point_index + 1) % points_count
        delta_x = polygon_points[next_index][0] - polygon_points[point_index][0]
        delta_y = polygon_points[next_index][1] - polygon_points[point_index][1]
        perimeter_value += math.hypot(delta_x, delta_y)

    return perimeter_value


def is_point_on_segment(point_a, point_b, point_to_check):
    cross_value = cross_product(point_a, point_b, point_to_check)
    if abs(cross_value) > epsilon_value:
        return False

    minimum_x = min(point_a[0], point_b[0]) - epsilon_value
    maximum_x = max(point_a[0], point_b[0]) + epsilon_value
    minimum_y = min(point_a[1], point_b[1]) - epsilon_value
    maximum_y = max(point_a[1], point_b[1]) + epsilon_value

    return (
        minimum_x <= point_to_check[0] <= maximum_x
        and minimum_y <= point_to_check[1] <= maximum_y
    )


def point_in_convex_polygon(point_to_check, polygon_points):
    if len(polygon_points) < 3:
        return False

    has_positive = False
    has_negative = False
    points_count = len(polygon_points)

    for point_index in range(points_count):
        next_index = (point_index + 1) % points_count
        current_cross = cross_product(
            polygon_points[point_index],
            polygon_points[next_index],
            point_to_check
        )

        if abs(current_cross) <= epsilon_value:
            if is_point_on_segment(
                polygon_points[point_index],
                polygon_points[next_index],
                point_to_check
            ):
                return True
        elif current_cross > 0:
            has_positive = True
        else:
            has_negative = True

        if has_positive and has_negative:
            return False

    return True


def remove_duplicate_points(points):
    unique_points = []
    used_points = set()

    for point_value in points:
        rounded_point = (round(point_value[0], 10), round(point_value[1], 10))
        if rounded_point not in used_points:
            used_points.add(rounded_point)
            unique_points.append((point_value[0], point_value[1]))

    return unique_points


def graham_convex_hull(point_set):
    unique_points = remove_duplicate_points(point_set)

    if len(unique_points) <= 1:
        return unique_points

    start_point = min(unique_points, key=lambda point_value: (point_value[1], point_value[0]))

    def polar_angle_and_distance(point_value):
        angle_value = math.atan2(
            point_value[1] - start_point[1],
            point_value[0] - start_point[0]
        )
        distance_value = distance_squared(start_point, point_value)
        return angle_value, distance_value

    sorted_points = sorted(unique_points, key=polar_angle_and_distance)

    filtered_points = []
    for point_value in sorted_points:
        while (
            len(filtered_points) > 0
            and abs(
                math.atan2(
                    point_value[1] - start_point[1],
                    point_value[0] - start_point[0]
                )
                - math.atan2(
                    filtered_points[-1][1] - start_point[1],
                    filtered_points[-1][0] - start_point[0]
                )
            ) <= epsilon_value
        ):
            if distance_squared(start_point, point_value) >= distance_squared(start_point, filtered_points[-1]):
                filtered_points.pop()
            else:
                break
        else:
            filtered_points.append(point_value)
            continue

        if (
            len(filtered_points) == 0
            or filtered_points[-1] != point_value
        ):
            filtered_points.append(point_value)

    if len(filtered_points) < 3:
        return filtered_points

    hull_stack = [filtered_points[0], filtered_points[1]]

    for point_value in filtered_points[2:]:
        while (
            len(hull_stack) >= 2
            and cross_product(hull_stack[-2], hull_stack[-1], point_value) <= epsilon_value
        ):
            hull_stack.pop()
        hull_stack.append(point_value)

    return hull_stack


def jarvis_convex_hull(point_set):
    unique_points = remove_duplicate_points(point_set)

    if len(unique_points) <= 1:
        return unique_points

    start_point = min(unique_points, key=lambda point_value: (point_value[1], point_value[0]))
    hull_points = []
    current_point = start_point

    while True:
        hull_points.append(current_point)
        candidate_point = None

        for point_value in unique_points:
            if point_value == current_point:
                continue
            candidate_point = point_value
            break

        for point_value in unique_points:
            if point_value == current_point or point_value == candidate_point:
                continue

            orientation_value = cross_product(current_point, candidate_point, point_value)

            if orientation_value > epsilon_value:
                candidate_point = point_value
            elif abs(orientation_value) <= epsilon_value:
                if distance_squared(current_point, point_value) > distance_squared(current_point, candidate_point):
                    candidate_point = point_value

        current_point = candidate_point

        if current_point == start_point:
            break

    return hull_points


def line_intersection(point_a1, point_a2, point_b1, point_b2):
    x1, y1 = point_a1
    x2, y2 = point_a2
    x3, y3 = point_b1
    x4, y4 = point_b2

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denominator) <= epsilon_value:
        return None

    determinant_1 = x1 * y2 - y1 * x2
    determinant_2 = x3 * y4 - y3 * x4

    intersection_x = (
        determinant_1 * (x3 - x4) - (x1 - x2) * determinant_2
    ) / denominator
    intersection_y = (
        determinant_1 * (y3 - y4) - (y1 - y2) * determinant_2
    ) / denominator

    return (intersection_x, intersection_y)


def inside_edge(point_value, edge_start, edge_end):
    return cross_product(edge_start, edge_end, point_value) >= -epsilon_value


def polygon_intersection(subject_polygon, clip_polygon):
    if len(subject_polygon) < 3 or len(clip_polygon) < 3:
        return []

    output_polygon = subject_polygon[:]

    for clip_index in range(len(clip_polygon)):
        clip_edge_start = clip_polygon[clip_index]
        clip_edge_end = clip_polygon[(clip_index + 1) % len(clip_polygon)]

        input_polygon = output_polygon[:]
        output_polygon = []

        if len(input_polygon) == 0:
            break

        previous_point = input_polygon[-1]

        for current_point in input_polygon:
            current_inside = inside_edge(current_point, clip_edge_start, clip_edge_end)
            previous_inside = inside_edge(previous_point, clip_edge_start, clip_edge_end)

            if current_inside:
                if not previous_inside:
                    intersection_point = line_intersection(
                        previous_point,
                        current_point,
                        clip_edge_start,
                        clip_edge_end
                    )
                    if intersection_point is not None:
                        output_polygon.append(intersection_point)
                output_polygon.append(current_point)
            elif previous_inside:
                intersection_point = line_intersection(
                    previous_point,
                    current_point,
                    clip_edge_start,
                    clip_edge_end
                )
                if intersection_point is not None:
                    output_polygon.append(intersection_point)

            previous_point = current_point

    return remove_duplicate_points(output_polygon)


def points_inside_polygon(point_set, polygon_points):
    return [
        point_value
        for point_value in point_set
        if point_in_convex_polygon(point_value, polygon_points)
    ]


def print_points(points_title, points):
    print(points_title)
    for point_index, point_value in enumerate(points, start=1):
        print(f"{point_index:2d}: ({point_value[0]:.3f}, {point_value[1]:.3f})")
    print()


def draw_polygon(polygon_points, line_style, label_text):
    if len(polygon_points) == 0:
        return

    closed_polygon = polygon_points + [polygon_points[0]]
    x_values = [point_value[0] for point_value in closed_polygon]
    y_values = [point_value[1] for point_value in closed_polygon]
    plt.plot(x_values, y_values, line_style, linewidth=2, label=label_text)


def main():
    point_set_g = [
        (1.0, 1.0),
        (2.0, 2.5),
        (3.0, 1.2),
        (4.0, 3.0),
        (5.2, 1.1),
        (6.5, 2.0),
        (7.5, 4.0),
        (6.8, 6.0),
        (5.5, 7.2),
        (3.8, 7.8),
        (2.0, 6.5),
        (1.2, 4.5),
        (2.5, 4.0),
        (3.5, 5.0),
        (4.8, 4.7),
        (5.8, 5.2),
        (3.0, 3.3),
        (4.5, 2.5)
    ]

    point_set_d = [
        (4.0, 0.8),
        (5.5, 1.5),
        (7.2, 1.0),
        (8.5, 2.8),
        (9.0, 4.5),
        (8.2, 6.8),
        (6.5, 8.0),
        (4.8, 8.5),
        (3.0, 7.2),
        (2.2, 5.4),
        (2.5, 3.5),
        (3.2, 2.0),
        (4.5, 4.2),
        (5.8, 3.8),
        (6.8, 5.0),
        (5.2, 6.2),
        (7.0, 6.0),
        (4.0, 6.0)
    ]

    convex_hull_g = graham_convex_hull(point_set_g)
    convex_hull_d = jarvis_convex_hull(point_set_d)

    perimeter_g = polygon_perimeter(convex_hull_g)
    area_g = polygon_area(convex_hull_g)

    perimeter_d = polygon_perimeter(convex_hull_d)
    area_d = polygon_area(convex_hull_d)

    intersection_polygon = polygon_intersection(convex_hull_g, convex_hull_d)

    inner_g_points = points_inside_polygon(point_set_g, intersection_polygon) if len(intersection_polygon) >= 3 else []
    inner_d_points = points_inside_polygon(point_set_d, intersection_polygon) if len(intersection_polygon) >= 3 else []

    print_points("Множество G:", point_set_g)
    print_points("Множество D:", point_set_d)

    print_points("Выпуклая оболочка 𝒢 = conv(G), алгоритм Грэхема:", convex_hull_g)
    print(f"Периметр 𝒢 = {perimeter_g:.3f}")
    print(f"Площадь 𝒢 = {area_g:.3f}")
    print()

    print_points("Выпуклая оболочка 𝒟 = conv(D), алгоритм Джарвиса:", convex_hull_d)
    print(f"Периметр 𝒟 = {perimeter_d:.3f}")
    print(f"Площадь 𝒟 = {area_d:.3f}")
    print()

    if len(intersection_polygon) >= 3:
        print_points("Пересечение 𝒫 = 𝒢 ∩ 𝒟:", intersection_polygon)
        print(f"Периметр 𝒫 = {polygon_perimeter(intersection_polygon):.3f}")
        print(f"Площадь 𝒫 = {polygon_area(intersection_polygon):.3f}")
        print()
    else:
        print("Пересечение 𝒫 = 𝒢 ∩ 𝒟 пусто или вырождено.")
        print()

    print_points("Точки множества G, лежащие внутри 𝒫:", inner_g_points)
    print_points("Точки множества D, лежащие внутри 𝒫:", inner_d_points)

    plt.figure(figsize=(10, 8))

    g_x_values = [point_value[0] for point_value in point_set_g]
    g_y_values = [point_value[1] for point_value in point_set_g]
    d_x_values = [point_value[0] for point_value in point_set_d]
    d_y_values = [point_value[1] for point_value in point_set_d]

    plt.scatter(g_x_values, g_y_values, marker='o', s=50, label='Точки G')
    plt.scatter(d_x_values, d_y_values, marker='s', s=50, label='Точки D')

    draw_polygon(convex_hull_g, '-', 'Оболочка 𝒢 (Грэхем)')
    draw_polygon(convex_hull_d, '--', 'Оболочка 𝒟 (Джарвис)')

    if len(intersection_polygon) >= 3:
        intersection_x_values = [point_value[0] for point_value in intersection_polygon]
        intersection_y_values = [point_value[1] for point_value in intersection_polygon]
        plt.fill(intersection_x_values, intersection_y_values, alpha=0.25, label='Пересечение 𝒫')

    if len(inner_g_points) > 0:
        inner_g_x_values = [point_value[0] for point_value in inner_g_points]
        inner_g_y_values = [point_value[1] for point_value in inner_g_points]
        plt.scatter(inner_g_x_values, inner_g_y_values, marker='*', s=180, label='Точки G внутри 𝒫')

    if len(inner_d_points) > 0:
        inner_d_x_values = [point_value[0] for point_value in inner_d_points]
        inner_d_y_values = [point_value[1] for point_value in inner_d_points]
        plt.scatter(inner_d_x_values, inner_d_y_values, marker='P', s=130, label='Точки D внутри 𝒫')

    plt.title('Задание 4.1. Выпуклые оболочки, их пересечение и внутренние точки')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()