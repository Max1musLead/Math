import math
import matplotlib.pyplot as plt

polygon_vertices = [
    (0, 0),
    (8, 0),
    (8, 2),
    (5, 2),
    (5, 4),
    (8, 4),
    (8, 8),
    (6, 8),
    (0, 8),
    (0, 4),
    (3, 4)
]

points_to_check = [
    (1, 1),
    (4, 1),
    (6, 1),
    (6, 3),
    (2, 6),
    (6, 6),
    (9, 3),
    (3, 4),
    (4, 8),
    (0, 3)
]

epsilon_value = 1e-9

def cross_product(point_a, point_b, point_c):
    return (
        (point_b[0] - point_a[0]) * (point_c[1] - point_a[1])
        - (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])
    )

def is_point_on_segment(point, segment_start, segment_end):
    if abs(cross_product(segment_start, segment_end, point)) > epsilon_value:
        return False

    min_x = min(segment_start[0], segment_end[0]) - epsilon_value
    max_x = max(segment_start[0], segment_end[0]) + epsilon_value
    min_y = min(segment_start[1], segment_end[1]) - epsilon_value
    max_y = max(segment_start[1], segment_end[1]) + epsilon_value

    return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y

def determine_position_by_angle_method(point, polygon):
    total_angle = 0.0
    vertices_count = len(polygon)

    for vertex_index in range(vertices_count):
        current_vertex = polygon[vertex_index]
        next_vertex = polygon[(vertex_index + 1) % vertices_count]

        if is_point_on_segment(point, current_vertex, next_vertex):
            return "на границе"

        vector_1_x = current_vertex[0] - point[0]
        vector_1_y = current_vertex[1] - point[1]
        vector_2_x = next_vertex[0] - point[0]
        vector_2_y = next_vertex[1] - point[1]

        current_cross_product = vector_1_x * vector_2_y - vector_1_y * vector_2_x
        current_dot_product = vector_1_x * vector_2_x + vector_1_y * vector_2_y

        total_angle += math.atan2(current_cross_product, current_dot_product)

    if abs(abs(total_angle) - 2 * math.pi) < 1e-6:
        return "внутри"

    return "снаружи"

def determine_position_by_ray_method(point, polygon):
    point_x, point_y = point
    intersections_count = 0
    vertices_count = len(polygon)

    for vertex_index in range(vertices_count):
        point_a = polygon[vertex_index]
        point_b = polygon[(vertex_index + 1) % vertices_count]

        if is_point_on_segment(point, point_a, point_b):
            return "на границе"

        point_a_x, point_a_y = point_a
        point_b_x, point_b_y = point_b

        if point_a_y > point_b_y:
            point_a_x, point_b_x = point_b_x, point_a_x
            point_a_y, point_b_y = point_b_y, point_a_y

        if abs(point_a_y - point_b_y) < epsilon_value:
            continue

        if not (point_a_y < point_y <= point_b_y):
            continue

        intersection_x = point_a_x + (
            (point_y - point_a_y) * (point_b_x - point_a_x)
            / (point_b_y - point_a_y)
        )

        if intersection_x > point_x + epsilon_value:
            intersections_count += 1

    if intersections_count % 2 == 1:
        return "внутри"

    return "снаружи"


def main():
    print("Вершины невыпуклого многоугольника:")
    for vertex_index, vertex in enumerate(polygon_vertices, start=1):
        print(f"P{vertex_index} = {vertex}")

    print("\nПоложение точек относительно многоугольника:\n")

    point_positions_for_plot = []

    for point_index, point in enumerate(points_to_check, start=1):
        angle_method_result = determine_position_by_angle_method(point, polygon_vertices)
        ray_method_result = determine_position_by_ray_method(point, polygon_vertices)

        point_positions_for_plot.append(ray_method_result)

        print(f"M{point_index} = {point}")
        print(f"  Угловой метод: {angle_method_result}")
        print(f"  Лучевой метод: {ray_method_result}")
        print()

    polygon_x_coordinates = [vertex[0] for vertex in polygon_vertices] + [polygon_vertices[0][0]]
    polygon_y_coordinates = [vertex[1] for vertex in polygon_vertices] + [polygon_vertices[0][1]]

    plt.figure()

    plt.plot(polygon_x_coordinates, polygon_y_coordinates, linewidth=2)

    polygon_points_x = [vertex[0] for vertex in polygon_vertices]
    polygon_points_y = [vertex[1] for vertex in polygon_vertices]

    plt.scatter(polygon_points_x, polygon_points_y, marker='D', label='Вершины P')

    for point_index, point in enumerate(points_to_check, start=1):
        point_x, point_y = point
        point_position = point_positions_for_plot[point_index - 1]

        if point_position == "внутри":
            plt.scatter(point_x, point_y, marker='o')
        elif point_position == "снаружи":
            plt.scatter(point_x, point_y, marker='x')
        else:
            plt.scatter(point_x, point_y, marker='s')

        plt.text(point_x + 0.1, point_y + 0.1, f"M{point_index}")

    for vertex_index, vertex in enumerate(polygon_vertices, start=1):
        plt.text(vertex[0] + 0.1, vertex[1] + 0.1, f"P{vertex_index}")

    plt.title("Невыпуклый многоугольник и точки")
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()