from math import cos, sin, pi, tan
import numpy as np
from matplotlib import pyplot as plt

class F:
    def __init__(self, transformation_matrix):
        self.transformation_matrix = np.array(transformation_matrix, dtype=float)

    def apply(self, homogeneous_points):
        return self.transformation_matrix @ homogeneous_points


def main():
    radius = 1.0
    light_height = 1.0
    light_angle = pi / 2.5
    offset_x = -3.0
    offset_y = 4.0
    circle_height = 2.0

    horizontal_shift = light_height * tan(light_angle)
    center_x = offset_x - horizontal_shift
    center_y = offset_y + circle_height

    semi_axis_x = radius
    semi_axis_y = radius * cos(light_angle)

    print(
        "Параметры: R =", radius,
        ", h =", light_height,
        ", угол =", light_angle,
        ", смещение = (", offset_x, ",", offset_y, ")"
    )
    print("Центр окружности:", (center_x, center_y))
    print("Полуоси эллипса: a =", semi_axis_x, ", b =", semi_axis_y)

    circle_matrix = np.array([
        [1.0, 0.0, -center_x],
        [0.0, 1.0, -center_y],
        [-center_x, -center_y, center_x * center_x + center_y * center_y - radius * radius]
    ])

    print("\nb) Матрица окружности в однородных координатах:")
    print("(x - cx)^2 + (y - cy)^2 - R^2 = 0")
    print(circle_matrix)

    transformation_matrix = np.array([
        [1.0, 0.0, -center_x],
        [0.0, cos(light_angle), -cos(light_angle) * center_y],
        [0.0, 0.0, 1.0]
    ])

    print("\nc) Матрица аффинного преобразования F:")
    print(transformation_matrix)

    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)
    ellipse_matrix = inverse_transformation_matrix.T @ circle_matrix @ inverse_transformation_matrix

    print("\nd) Матрица эллипса Q' = (F^(-1))^T Q F^(-1):")
    print(ellipse_matrix)

    q11 = float(ellipse_matrix[0, 0])
    q12 = float(ellipse_matrix[0, 1])
    q13 = float(ellipse_matrix[0, 2])
    q22 = float(ellipse_matrix[1, 1])
    q23 = float(ellipse_matrix[1, 2])
    q33 = float(ellipse_matrix[2, 2])

    print("\nДекартово уравнение эллипса:")
    print(
        f"{q11:.4g}*x² + {2 * q12:.4g}*xy + {q22:.4g}*y² + "
        f"{2 * q13:.4g}*x + {2 * q23:.4g}*y + {q33:.4g} = 0"
    )

    point_count = 200
    parameter_values = np.linspace(0, 2 * pi, point_count)

    circle_points = np.array([
        [center_x + radius * cos(parameter_value) for parameter_value in parameter_values],
        [center_y + radius * sin(parameter_value) for parameter_value in parameter_values],
        [1.0] * point_count
    ])

    affine_transformation = F(transformation_matrix)
    ellipse_points = affine_transformation.apply(circle_points)
    ellipse_array = np.array(ellipse_points)

    figure, axes = plt.subplots(figsize=(8, 6))

    axes.plot(
        ellipse_array[0, :] / ellipse_array[2, :],
        ellipse_array[1, :] / ellipse_array[2, :],
        "b-",
        linewidth=1.5,
        label="Проекция (эллипс тени)"
    )

    axes.plot(
        center_x + radius * np.cos(parameter_values),
        center_y + radius * np.sin(parameter_values),
        "g-",
        linewidth=2,
        label="Круг (шар, сверху)"
    )

    left_circle_x = center_x - radius
    left_circle_y = center_y
    right_circle_x = center_x + radius
    right_circle_y = center_y

    left_ellipse_x = -semi_axis_x
    left_ellipse_y = 0.0
    right_ellipse_x = semi_axis_x
    right_ellipse_y = 0.0

    axes.plot(
        [left_circle_x, left_ellipse_x],
        [left_circle_y, left_ellipse_y],
        "k--",
        linewidth=1
    )
    axes.plot(
        [right_circle_x, right_ellipse_x],
        [right_circle_y, right_ellipse_y],
        "k--",
        linewidth=1,
        label="Границы луча света"
    )

    axes.fill(
        semi_axis_x * np.cos(parameter_values),
        semi_axis_y * np.sin(parameter_values),
        color="gray",
        alpha=0.3,
        zorder=0,
        label="Тень"
    )

    axes.set_aspect("equal", adjustable="box")
    axes.axhline(0, color="k", linewidth=0.5)
    axes.axvline(0, color="k", linewidth=0.5)
    axes.grid(True)
    axes.legend()
    axes.set_title("Преобразование квадрики\nОкружность (шар) → Эллипс (тень)")

    plt.tight_layout()
    plt.show()

    geometric_x = ellipse_array[0, :] / ellipse_array[2, :]
    geometric_y = ellipse_array[1, :] / ellipse_array[2, :]
    analytic_x = semi_axis_x * np.cos(parameter_values)
    analytic_y = semi_axis_y * np.sin(parameter_values)

    difference = np.sqrt((geometric_x - analytic_x) ** 2 + (geometric_y - analytic_y) ** 2)
    maximum_difference = np.max(difference)

    print("\ne) Проверка совпадения геометрического и аналитического образов:")
    if maximum_difference < 1e-10:
        print("Образы совпадают.")
    else:
        print("Есть расхождение.")
        print("Максимальное отклонение =", maximum_difference)

def main2():
    light_angle = pi / 2.5
    offset_x = -3.0
    offset_y = 4.0
    parabola_center_y = 2.0

    vertex_x = offset_x
    vertex_y = offset_y + parabola_center_y

    parabola_parameter = 0.8

    print("Параметры параболы:")
    print("Вершина параболы:", (vertex_x, vertex_y))
    print("p =", parabola_parameter)
    print("Угол =", light_angle)

    parabola_matrix = np.array([
        [0.0, 0.0, -parabola_parameter],
        [0.0, 1.0, -vertex_y],
        [-parabola_parameter, -vertex_y, vertex_y * vertex_y + 2.0 * parabola_parameter * vertex_x]
    ])

    print("\nМатрица параболы в однородных координатах:")
    print(parabola_matrix)

    transformation_matrix = np.array([
        [1.0, 0.0, -vertex_x],
        [0.0, np.cos(light_angle), -np.cos(light_angle) * vertex_y],
        [0.0, 0.0, 1.0]
    ])

    print("\nМатрица аффинного преобразования F:")
    print(transformation_matrix)

    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)
    transformed_parabola_matrix = inverse_transformation_matrix.T @ parabola_matrix @ inverse_transformation_matrix

    print("\nМатрица образа параболы:")
    print(transformed_parabola_matrix)

    q11 = float(transformed_parabola_matrix[0, 0])
    q12 = float(transformed_parabola_matrix[0, 1])
    q13 = float(transformed_parabola_matrix[0, 2])
    q22 = float(transformed_parabola_matrix[1, 1])
    q23 = float(transformed_parabola_matrix[1, 2])
    q33 = float(transformed_parabola_matrix[2, 2])

    print("\nДекартово уравнение образа:")
    print(
        f"{q11:.4g}*x² + {2 * q12:.4g}*xy + {q22:.4g}*y² + "
        f"{2 * q13:.4g}*x + {2 * q23:.4g}*y + {q33:.4g} = 0"
    )

    point_count = 400
    parameter_values = np.linspace(-3.0, 3.0, point_count)

    source_x = vertex_x + (parameter_values ** 2) / (2.0 * parabola_parameter)
    source_y = vertex_y + parameter_values

    source_points = np.array([
        source_x,
        source_y,
        np.ones(point_count)
    ])

    affine_transformation = F(transformation_matrix)
    transformed_points = affine_transformation.apply(source_points)

    transformed_x = transformed_points[0, :] / transformed_points[2, :]
    transformed_y = transformed_points[1, :] / transformed_points[2, :]

    figure, axes = plt.subplots(figsize=(8, 6))

    left_parameter = 3
    right_parameter = 0

    left_parabola_x = vertex_x + (left_parameter ** 2) / (2.0 * parabola_parameter)
    left_parabola_y = vertex_y + left_parameter

    right_parabola_x = vertex_x + (right_parameter ** 2) / (2.0 * parabola_parameter)
    right_parabola_y = vertex_y + right_parameter

    left_point = np.array([[left_parabola_x], [left_parabola_y], [1.0]])
    right_point = np.array([[right_parabola_x], [right_parabola_y], [1.0]])

    left_transformed = affine_transformation.apply(left_point)
    right_transformed = affine_transformation.apply(right_point)

    left_transformed_x = left_transformed[0, 0] / left_transformed[2, 0]
    left_transformed_y = left_transformed[1, 0] / left_transformed[2, 0]

    right_transformed_x = right_transformed[0, 0] / right_transformed[2, 0]
    right_transformed_y = right_transformed[1, 0] / right_transformed[2, 0]

    axes.plot(
        [left_parabola_x, left_transformed_x],
        [left_parabola_y, left_transformed_y],
        "k--",
        linewidth=1
    )

    axes.plot(
        [right_parabola_x, right_transformed_x],
        [right_parabola_y, right_transformed_y],
        "k--",
        linewidth=1,
        label="Границы луча света"
    )

    axes.plot(
        source_x,
        source_y,
        "g-",
        linewidth=2,
        label="Парабола"
    )

    axes.plot(
        transformed_x,
        transformed_y,
        "b-",
        linewidth=2,
        label="Образ параболы"
    )

    axes.scatter(
        [vertex_x],
        [vertex_y],
        color="red",
        s=50,
        label="Вершина параболы"
    )

    axes.set_aspect("equal", adjustable="box")
    axes.axhline(0, color="k", linewidth=0.5)
    axes.axvline(0, color="k", linewidth=0.5)
    axes.grid(True)
    axes.legend()
    axes.set_title("Преобразование параболы аффинным преобразованием")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main2()