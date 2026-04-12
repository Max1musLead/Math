from math import cos, sin, pi
import numpy as np
from matplotlib import pyplot as plt


class Transformation:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=float)

    def apply_to_points(self, points):
        homogeneous_points = np.column_stack(
            [points[:, 0], points[:, 1], np.ones(len(points))]
        )
        transformed_points = (self.matrix @ homogeneous_points.T).T
        return transformed_points[:, :2]


def cubic_bezier(control_points, parameter_values):
    point_0 = np.array(control_points[0], dtype=float)
    point_1 = np.array(control_points[1], dtype=float)
    point_2 = np.array(control_points[2], dtype=float)
    point_3 = np.array(control_points[3], dtype=float)

    curve_points = []
    for parameter_t in parameter_values:
        curve_point = (
            (1 - parameter_t) ** 3 * point_0
            + 3 * parameter_t * (1 - parameter_t) ** 2 * point_1
            + 3 * parameter_t ** 2 * (1 - parameter_t) * point_2
            + parameter_t ** 3 * point_3
        )
        curve_points.append(curve_point)

    return np.array(curve_points)


def quadratic_bezier(control_points, parameter_values):
    point_0 = np.array(control_points[0], dtype=float)
    point_1 = np.array(control_points[1], dtype=float)
    point_2 = np.array(control_points[2], dtype=float)

    curve_points = []
    for parameter_t in parameter_values:
        curve_point = (
            (1 - parameter_t) ** 2 * point_0
            + 2 * parameter_t * (1 - parameter_t) * point_1
            + parameter_t ** 2 * point_2
        )
        curve_points.append(curve_point)

    return np.array(curve_points)

def plot_control_polygon(axis, control_points, label_text, color):
    control_points_array = np.array(control_points, dtype=float)
    axis.plot(
        control_points_array[:, 0],
        control_points_array[:, 1],
        "o--",
        color=color,
        linewidth=1.2,
        markersize=5,
        label=label_text
    )


def main():
    square_points = {
        "A": np.array([0.0, 0.0]),
        "B": np.array([2.0, 3.0]),
        "C": np.array([-2.0, 3.0]),
        "D": np.array([1.0, 0.0]),
    }

    parameter_values = np.linspace(0.0, 1.0, 500)

    bezier_points_a = [
        square_points["A"],
        square_points["B"],
        square_points["C"],
        square_points["D"],
    ]

    bezier_points_b = [
        square_points["A"],
        square_points["C"],
        square_points["B"],
        square_points["D"],
    ]

    bezier_curve_a = cubic_bezier(bezier_points_a, parameter_values)

    angle_phi = pi
    shear_factor_k = 0.6
    translation_x = 2.0
    translation_y = 1.0

    rotation_matrix = np.array([
        [cos(angle_phi), -sin(angle_phi), 0.0],
        [sin(angle_phi),  cos(angle_phi), 0.0],
        [0.0,             0.0,            1.0],
    ])

    shear_matrix = np.array([
        [1.0, shear_factor_k, 0.0],
        [0.0, 1.0,            0.0],
        [0.0, 0.0,            1.0],
    ])

    translation_matrix = np.array([
        [1.0, 0.0, translation_x],
        [0.0, 1.0, translation_y],
        [0.0, 0.0, 1.0],
    ])

    total_transformation_matrix = translation_matrix @ shear_matrix @ rotation_matrix
    transformation = Transformation(total_transformation_matrix)
    transformed_bezier_curve_a = transformation.apply_to_points(bezier_curve_a)
    transformed_control_polygon_a = transformation.apply_to_points(np.array(bezier_points_a))

    print("Классификация перестановок:")
    print("a) Простая, выпуклая вправо: A-B-C-D.")
    print("b) С точкой возврата, без самопересечений: A-C-B-D.")
    print("c) С самопересечениями: таких перестановок нет.")
    print()
    print("Итоговая матрица F = T * Shx * R:")
    print(total_transformation_matrix)

    figure_task_1, axis_task_1 = plt.subplots(figsize=(10, 8))

    plot_control_polygon(axis_task_1, bezier_points_a, "Опорный многоугольник A-B-C-D", "gray")
    axis_task_1.plot(
        bezier_curve_a[:, 0],
        bezier_curve_a[:, 1],
        linewidth=2.5,
        label="a) A-B-C-D: простая, выпуклая вправо"
    )

    shift_value = 1.5

    bezier_points_b_shifted = [
        point + np.array([shift_value, 0.0])
        for point in bezier_points_b
    ]

    bezier_curve_b_shifted = cubic_bezier(bezier_points_b_shifted, parameter_values)

    plot_control_polygon(axis_task_1, bezier_points_b_shifted, "Опорный многоугольник A-C-B-D", "gray")
    axis_task_1.plot(
        bezier_curve_b_shifted[:, 0],
        bezier_curve_b_shifted[:, 1],
        linewidth=2.5,
        label="b) A-C-B-D: с точкой возврата"
    )

    axis_task_1.plot(
        transformed_bezier_curve_a[:, 0],
        transformed_bezier_curve_a[:, 1],
        linewidth=2.5,
        label="F(B(t)) для кривой из пункта a)"
    )
    axis_task_1.plot(
        transformed_control_polygon_a[:, 0],
        transformed_control_polygon_a[:, 1],
        "o--",
        linewidth=1.2,
        markersize=5,
        label="Преобразованный опорный многоугольник"
    )

    for point_name, point_coordinates in square_points.items():
        axis_task_1.text(
            point_coordinates[0] + 0.03,
            point_coordinates[1] + 0.03,
            point_name,
            fontsize=11
        )

    axis_task_1.set_title("Задание I. Кубические кривые Безье по вершинам квадрата")
    axis_task_1.set_aspect("equal")
    axis_task_1.grid(True)
    axis_task_1.legend()

    figure_task_2, axis_task_2 = plt.subplots(figsize=(10, 8))

    # Панцирь
    shell_upper = cubic_bezier(
        [(-2.8, 0.0), (-2.8, 3.25), (2.8, 3.25), (2.8, 0.0)],
        parameter_values
    )

    shell_lower = cubic_bezier(
        [(2.8, 0.0), (2.8, -3.25), (-2.8, -3.25), (-2.8, 0.0)],
        parameter_values
    )

    # Голова
    head_upper = quadratic_bezier(
        [(2.8, 0.5), (3.8, 1.2), (4.5, 0.0)],
        parameter_values
    )
    head_lower = quadratic_bezier(
        [(2.8, -0.5), (3.8, -1.2), (4.5, 0.0)],
        parameter_values
    )

    # Хвост
    tail_upper = quadratic_bezier(
        [(-2.8, 0.3), (-3.6, 0.7), (-4.0, 0.0)],
        parameter_values
    )
    tail_lower = quadratic_bezier(
        [(-2.8, -0.3), (-3.6, -0.7), (-4.0, 0.0)],
        parameter_values
    )

    # Лапы
    front_leg_lower_outer = cubic_bezier(
        [(1.8, -1.8), (2.15, -2.1), (2.35, -2.45), (2.2, -3.0)],
        parameter_values
    )
    front_leg_lower_inner = cubic_bezier(
        [(2.2, -3.0), (1.95, -2.95), (1.75, -2.35), (1.6, -1.9)],
        parameter_values
    )

    back_leg_lower_outer = cubic_bezier(
        [(-1.8, -1.8), (-2.15, -2.1), (-2.35, -2.45), (-2.2, -3.0)],
        parameter_values
    )
    back_leg_lower_inner = cubic_bezier(
        [(-2.2, -3.0), (-1.95, -2.95), (-1.75, -2.35), (-1.6, -1.9)],
        parameter_values
    )

    front_leg_upper_outer = cubic_bezier(
        [(1.8, 1.8), (2.15, 2.1), (2.35, 2.45), (2.2, 3.0)],
        parameter_values
    )
    front_leg_upper_inner = cubic_bezier(
        [(2.2, 3.0), (1.95, 2.95), (1.75, 2.35), (1.6, 1.9)],
        parameter_values
    )

    back_leg_upper_outer = cubic_bezier(
        [(-1.8, 1.8), (-2.15, 2.1), (-2.35, 2.45), (-2.2, 3.0)],
        parameter_values
    )
    back_leg_upper_inner = cubic_bezier(
        [(-2.2, 3.0), (-1.95, 2.95), (-1.75, 2.35), (-1.6, 1.9)],
        parameter_values
    )

    # Узор
    pattern_1 = quadratic_bezier([(-1.5, 0), (0, 1), (1.5, 0)], parameter_values)
    pattern_2 = quadratic_bezier([(-1.5, 0), (0, -1), (1.5, 0)], parameter_values)

    # Глаза
    eye_1 = (4.0, 0.2)
    eye_2 = (4.0, -0.2)

    axis_task_2.plot(shell_upper[:, 0], shell_upper[:, 1], linewidth=2.7, label="Панцирь (сплайн из Безье)")
    axis_task_2.plot(shell_lower[:, 0], shell_lower[:, 1], linewidth=2.7)

    axis_task_2.plot(head_upper[:, 0], head_upper[:, 1], linewidth=2.2, label="Голова/хвост (Безье 2)")
    axis_task_2.plot(head_lower[:, 0], head_lower[:, 1], linewidth=2.2)
    axis_task_2.plot(tail_upper[:, 0], tail_upper[:, 1], linewidth=2.2)
    axis_task_2.plot(tail_lower[:, 0], tail_lower[:, 1], linewidth=2.2)

    axis_task_2.plot(
        front_leg_lower_outer[:, 0],
        front_leg_lower_outer[:, 1],
        linewidth=2.0,
        label="Лапы (Безье 3)"
    )
    axis_task_2.plot(
        front_leg_lower_inner[:, 0],
        front_leg_lower_inner[:, 1],
        linewidth=2.0
    )

    axis_task_2.plot(
        back_leg_lower_outer[:, 0],
        back_leg_lower_outer[:, 1],
        linewidth=2.0
    )
    axis_task_2.plot(
        back_leg_lower_inner[:, 0],
        back_leg_lower_inner[:, 1],
        linewidth=2.0
    )

    axis_task_2.plot(
        front_leg_upper_outer[:, 0],
        front_leg_upper_outer[:, 1],
        linewidth=2.0
    )
    axis_task_2.plot(
        front_leg_upper_inner[:, 0],
        front_leg_upper_inner[:, 1],
        linewidth=2.0
    )

    axis_task_2.plot(
        back_leg_upper_outer[:, 0],
        back_leg_upper_outer[:, 1],
        linewidth=2.0
    )
    axis_task_2.plot(
        back_leg_upper_inner[:, 0],
        back_leg_upper_inner[:, 1],
        linewidth=2.0
    )

    axis_task_2.plot(pattern_1[:, 0], pattern_1[:, 1], linewidth=1.5)
    axis_task_2.plot(pattern_2[:, 0], pattern_2[:, 1], linewidth=1.5)

    # глаза
    axis_task_2.plot(*eye_1, "o", markersize=4)
    axis_task_2.plot(*eye_2, "o", markersize=4)

    axis_task_2.set_title("Контур черепашки")
    axis_task_2.set_aspect("equal")
    axis_task_2.grid(True)

    axis_task_2.legend(loc="lower right")

    def draw_points(points):
        points_array = np.array(points, dtype=float)
        axis_task_2.plot(
            points_array[:, 0],
            points_array[:, 1],
            "o--",
            linewidth=1.0,
            markersize=4
        )

    draw_points([(-2.8, 0.0), (-2.8, 3.25), (2.8, 3.25), (2.8, 0.0)])
    draw_points([(2.8, 0.0), (2.8, -3.25), (-2.8, -3.25), (-2.8, 0.0)])

    draw_points([(2.8, 0.5), (3.8, 1.2), (4.5, 0.0)])
    draw_points([(2.8, -0.5), (3.8, -1.2), (4.5, 0.0)])

    draw_points([(-2.8, 0.3), (-3.6, 0.7), (-4.0, 0.0)])
    draw_points([(-2.8, -0.3), (-3.6, -0.7), (-4.0, 0.0)])

    draw_points([(1.8, -1.8), (2.15, -2.1), (2.35, -2.45), (2.2, -3.0)])
    draw_points([(2.2, -3.0), (1.95, -2.95), (1.75, -2.35), (1.6, -1.9)])

    draw_points([(-1.8, -1.8), (-2.15, -2.1), (-2.35, -2.45), (-2.2, -3.0)])
    draw_points([(-2.2, -3.0), (-1.95, -2.95), (-1.75, -2.35), (-1.6, -1.9)])

    draw_points([(1.8, 1.8), (2.15, 2.1), (2.35, 2.45), (2.2, 3.0)])
    draw_points([(2.2, 3.0), (1.95, 2.95), (1.75, 2.35), (1.6, 1.9)])

    draw_points([(-1.8, 1.8), (-2.15, 2.1), (-2.35, 2.45), (-2.2, 3.0)])
    draw_points([(-2.2, 3.0), (-1.95, 2.95), (-1.75, 2.35), (-1.6, 1.9)])

    plt.show()


if __name__ == "__main__":
    main()