from math import sin, cos, pi, sqrt
import numpy as np
from matplotlib import pyplot as plt

EPSILON = 1e-9

def length(x_value, y_value):
    return sqrt(x_value * x_value + y_value * y_value)

def parabola(parameter_t):
    return parameter_t, parameter_t * parameter_t

def parabola_first(parameter_t):
    return 1.0, 2.0 * parameter_t

def parabola_second(parameter_t):
    return 0.0, 2.0

def cycloid(parameter_t, radius_value=1.0):
    return radius_value * (parameter_t - sin(parameter_t)), radius_value * (1.0 - cos(parameter_t))

def cycloid_first(parameter_t, radius_value=1.0):
    return radius_value * (1.0 - cos(parameter_t)), radius_value * sin(parameter_t)

def cycloid_second(parameter_t, radius_value=1.0):
    return radius_value * sin(parameter_t), radius_value * cos(parameter_t)

def unit_tangent(first_x, first_y):
    first_length = length(first_x, first_y)
    if first_length < EPSILON:
        return 0.0, 0.0
    return first_x / first_length, first_y / first_length

def unit_normal(first_x, first_y):
    tangent_x, tangent_y = unit_tangent(first_x, first_y)
    return -tangent_y, tangent_x

def curvature(first_x, first_y, second_x, second_y):
    speed_squared = first_x * first_x + first_y * first_y
    if speed_squared < EPSILON:
        return None

    numerator = first_x * second_y - first_y * second_x
    denominator = speed_squared ** 1.5

    if abs(denominator) < EPSILON:
        return None

    return numerator / denominator

def radius_of_curvature(curvature_value):
    if curvature_value is None or abs(curvature_value) < EPSILON:
        return None
    return 1.0 / abs(curvature_value)

def center_of_curvature(point_x, point_y, first_x, first_y, second_x, second_y):
    speed_squared = first_x * first_x + first_y * first_y
    determinant = first_x * second_y - first_y * second_x

    if speed_squared < EPSILON or abs(determinant) < EPSILON:
        return None

    factor = speed_squared / determinant
    center_x = point_x - factor * first_y
    center_y = point_y + factor * first_x
    return center_x, center_y

def build_evolute(point_function, first_function, second_function, parameter_values):
    evolute_x = []
    evolute_y = []

    for parameter_t in parameter_values:
        point_x, point_y = point_function(parameter_t)
        first_x, first_y = first_function(parameter_t)
        second_x, second_y = second_function(parameter_t)

        center = center_of_curvature(point_x, point_y, first_x, first_y, second_x, second_y)

        if center is None:
            evolute_x.append(np.nan)
            evolute_y.append(np.nan)
        else:
            center_x, center_y = center
            evolute_x.append(center_x)
            evolute_y.append(center_y)

    return np.array(evolute_x), np.array(evolute_y)

def print_info(curve_name, parameter_t0, point_x, point_y, tangent_x, tangent_y, normal_x, normal_y, curvature_value, radius_value):
    print(curve_name)
    print(f"t0 = {parameter_t0}")
    print(f"M0 = ({point_x:.6f}, {point_y:.6f})")
    print(f"Касательный единичный вектор = ({tangent_x:.6f}, {tangent_y:.6f})")
    print(f"Нормальный единичный вектор = ({normal_x:.6f}, {normal_y:.6f})")

    if curvature_value is None:
        print("Кривизна не определена")
    else:
        print(f"Кривизна k = {curvature_value:.6f}")

    if radius_value is None:
        print("Радиус кривизны не определён")
    else:
        print(f"Радиус кривизны R = {radius_value:.6f}")

    print()

def draw_curve(
    axes,
    title_text,
    curve_x,
    curve_y,
    evolute_x,
    evolute_y,
    point_x,
    point_y,
    tangent_x,
    tangent_y,
    normal_x,
    normal_y,
    curvature_center,
    radius_value,
    vector_scale
):
    axes.plot(curve_x, curve_y, label="Кривая", linewidth=2)
    axes.plot(evolute_x, evolute_y, "--", label="Эволюта", alpha=0.7)
    axes.plot(point_x, point_y, "ro", label="M0")

    # КАСАТЕЛЬНАЯ
    axes.annotate(
        "",
        xy=(point_x + tangent_x * vector_scale, point_y + tangent_y * vector_scale),
        xytext=(point_x, point_y),
        arrowprops=dict(arrowstyle="->", lw=3, color="black")
    )
    axes.text(
        point_x + tangent_x * vector_scale * 1.1,
        point_y + tangent_y * vector_scale * 1.1,
        "t",
        fontsize=12,
        color="black"
    )

    # НОРМАЛЬ
    axes.annotate(
        "",
        xy=(point_x + normal_x * vector_scale, point_y + normal_y * vector_scale),
        xytext=(point_x, point_y),
        arrowprops=dict(arrowstyle="->", lw=3, color="red")
    )
    axes.text(
        point_x + normal_x * vector_scale * 1.1,
        point_y + normal_y * vector_scale * 1.1,
        "n",
        fontsize=12,
        color="red"
    )

    if curvature_center is not None and radius_value is not None and np.isfinite(radius_value):
        center_x, center_y = curvature_center
        axes.plot(center_x, center_y, "go", label="Центр кривизны")

        angle_values = np.linspace(0.0, 2.0 * pi, 200)
        circle_x = center_x + radius_value * np.cos(angle_values)
        circle_y = center_y + radius_value * np.sin(angle_values)
        axes.plot(circle_x, circle_y, ":", label="Соприкасающаяся окружность")

    axes.set_title(title_text)

    axes.set_aspect("equal")
    axes.grid(True)
    axes.legend()

def main():
    # Параметры точек M0
    parabola_t0 = 1.0
    cycloid_t0 = pi / 2
    cycloid_radius = 1.0

    # Парабола вычисляем точку M0 и производные
    parabola_x0, parabola_y0 = parabola(parabola_t0)
    parabola_first_x0, parabola_first_y0 = parabola_first(parabola_t0)
    parabola_second_x0, parabola_second_y0 = parabola_second(parabola_t0)

    # Парабола вычисляем геометрические характеристики в точке
    parabola_tangent_x, parabola_tangent_y = unit_tangent(parabola_first_x0, parabola_first_y0)
    parabola_normal_x, parabola_normal_y = unit_normal(parabola_first_x0, parabola_first_y0)
    parabola_curvature = curvature(parabola_first_x0, parabola_first_y0, parabola_second_x0, parabola_second_y0)
    parabola_radius = radius_of_curvature(parabola_curvature)
    parabola_center = center_of_curvature(
        parabola_x0, parabola_y0,
        parabola_first_x0, parabola_first_y0,
        parabola_second_x0, parabola_second_y0
    )

    # Циклоида вычисляем точку M0 и производные
    cycloid_x0, cycloid_y0 = cycloid(cycloid_t0, cycloid_radius)
    cycloid_first_x0, cycloid_first_y0 = cycloid_first(cycloid_t0, cycloid_radius)
    cycloid_second_x0, cycloid_second_y0 = cycloid_second(cycloid_t0, cycloid_radius)

    # Циклоида вычисляем геометрические характеристики в точке
    cycloid_tangent_x, cycloid_tangent_y = unit_tangent(cycloid_first_x0, cycloid_first_y0)
    cycloid_normal_x, cycloid_normal_y = unit_normal(cycloid_first_x0, cycloid_first_y0)
    cycloid_curvature = curvature(cycloid_first_x0, cycloid_first_y0, cycloid_second_x0, cycloid_second_y0)
    cycloid_radius_value = radius_of_curvature(cycloid_curvature)
    cycloid_center = center_of_curvature(
        cycloid_x0, cycloid_y0,
        cycloid_first_x0, cycloid_first_y0,
        cycloid_second_x0, cycloid_second_y0
    )

    print_info(
        "=== C1: Парабола x=t, y=t^2 ===",
        parabola_t0,
        parabola_x0, parabola_y0,
        parabola_tangent_x, parabola_tangent_y,
        parabola_normal_x, parabola_normal_y,
        parabola_curvature,
        parabola_radius
    )

    print_info(
        "=== C2: Циклоида x=R(t-sin t), y=R(1-cos t) ===",
        cycloid_t0,
        cycloid_x0, cycloid_y0,
        cycloid_tangent_x, cycloid_tangent_y,
        cycloid_normal_x, cycloid_normal_y,
        cycloid_curvature,
        cycloid_radius_value
    )

    # Набор точек для построения кривых
    parabola_t_values = np.linspace(-3.0, 3.0, 500)
    parabola_x = np.array([parabola(parameter_t)[0] for parameter_t in parabola_t_values])
    parabola_y = np.array([parabola(parameter_t)[1] for parameter_t in parabola_t_values])

    cycloid_t_values = np.linspace(0.001, 4.0 * pi - 0.001, 700)
    cycloid_x = np.array([cycloid(parameter_t, cycloid_radius)[0] for parameter_t in cycloid_t_values])
    cycloid_y = np.array([cycloid(parameter_t, cycloid_radius)[1] for parameter_t in cycloid_t_values])

    # Эволюты
    parabola_evolute_x, parabola_evolute_y = build_evolute(
        parabola,
        parabola_first,
        parabola_second,
        parabola_t_values
    )

    cycloid_evolute_x, cycloid_evolute_y = build_evolute(
        lambda parameter_t: cycloid(parameter_t, cycloid_radius),
        lambda parameter_t: cycloid_first(parameter_t, cycloid_radius),
        lambda parameter_t: cycloid_second(parameter_t, cycloid_radius),
        cycloid_t_values
    )

    # График сами кривые и точки M0
    figure_preview, (axes_left, axes_right) = plt.subplots(1, 2, figsize=(12, 5))

    axes_left.plot(parabola_x, parabola_y)
    axes_left.plot(parabola_x0, parabola_y0, "ro", label="M0")
    axes_left.set_title("Парабола")
    axes_left.set_aspect("equal")
    axes_left.grid(True)
    axes_left.legend()

    axes_right.plot(cycloid_x, cycloid_y)
    axes_right.plot(cycloid_x0, cycloid_y0, "ro", label="M0")
    axes_right.set_title("Циклоида")
    axes_right.set_aspect("equal")
    axes_right.grid(True)
    axes_right.legend()

    plt.tight_layout()
    plt.show()

    # График параболы
    figure_parabola, axes_parabola = plt.subplots(figsize=(8, 6))
    draw_curve(
        axes_parabola,
        "C1: Парабола",
        parabola_x, parabola_y,
        parabola_evolute_x, parabola_evolute_y,
        parabola_x0, parabola_y0,
        parabola_tangent_x, parabola_tangent_y,
        parabola_normal_x, parabola_normal_y,
        parabola_center,
        parabola_radius,
        vector_scale=1.0
    )
    plt.tight_layout()
    plt.show()

    # График циклоиды
    figure_cycloid, axes_cycloid = plt.subplots(figsize=(8, 6))
    draw_curve(
        axes_cycloid,
        "C2: Циклоида",
        cycloid_x, cycloid_y,
        cycloid_evolute_x, cycloid_evolute_y,
        cycloid_x0, cycloid_y0,
        cycloid_tangent_x, cycloid_tangent_y,
        cycloid_normal_x, cycloid_normal_y,
        cycloid_center,
        cycloid_radius_value,
        vector_scale=0.6
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()