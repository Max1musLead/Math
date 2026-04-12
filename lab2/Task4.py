import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def natural_line_algorithm(x_start, y_start, x_end, y_end):
    raster_points = []

    delta_x = x_end - x_start
    delta_y = y_end - y_start

    if delta_x == 0 and delta_y == 0:
        return [(round(x_start), round(y_start))]

    if abs(delta_x) >= abs(delta_y):
        if x_start > x_end:
            x_start, y_start, x_end, y_end = x_end, y_end, x_start, y_start
            delta_x = x_end - x_start
            delta_y = y_end - y_start

        slope = delta_y / delta_x if delta_x != 0 else 0

        current_x = x_start
        while current_x <= x_end:
            current_y = y_start + slope * (current_x - x_start)
            raster_points.append((round(current_x), round(current_y)))
            current_x += 1
    else:
        if y_start > y_end:
            x_start, y_start, x_end, y_end = x_end, y_end, x_start, y_start
            delta_x = x_end - x_start
            delta_y = y_end - y_start

        inverse_slope = delta_x / delta_y if delta_y != 0 else 0

        current_y = y_start
        while current_y <= y_end:
            current_x = x_start + inverse_slope * (current_y - y_start)
            raster_points.append((round(current_x), round(current_y)))
            current_y += 1

    return remove_duplicate_points(raster_points)


def bresenham_line_algorithm(x_start, y_start, x_end, y_end):
    raster_points = []

    delta_x = abs(x_end - x_start)
    delta_y = abs(y_end - y_start)

    step_x = 1 if x_end >= x_start else -1
    step_y = 1 if y_end >= y_start else -1

    current_x = x_start
    current_y = y_start

    if delta_x >= delta_y:
        error_value = 2 * delta_y - delta_x

        for _ in range(delta_x + 1):
            raster_points.append((current_x, current_y))

            if error_value >= 0:
                current_y += step_y
                error_value += 2 * (delta_y - delta_x)
            else:
                error_value += 2 * delta_y

            current_x += step_x
    else:
        error_value = 2 * delta_x - delta_y

        for _ in range(delta_y + 1):
            raster_points.append((current_x, current_y))

            if error_value >= 0:
                current_x += step_x
                error_value += 2 * (delta_x - delta_y)
            else:
                error_value += 2 * delta_x

            current_y += step_y

    return raster_points


def remove_duplicate_points(points):
    unique_points = []
    visited_points = set()

    for point in points:
        if point not in visited_points:
            unique_points.append(point)
            visited_points.add(point)

    return unique_points


def translate_point(x_coordinate, y_coordinate, shift_x, shift_y):
    return x_coordinate + shift_x, y_coordinate + shift_y


def rotate_point(x_coordinate, y_coordinate, rotation_angle_degrees, rotation_center_x=0, rotation_center_y=0):
    rotation_angle_radians = math.radians(rotation_angle_degrees)

    shifted_x = x_coordinate - rotation_center_x
    shifted_y = y_coordinate - rotation_center_y

    rotated_x = shifted_x * math.cos(rotation_angle_radians) - shifted_y * math.sin(rotation_angle_radians)
    rotated_y = shifted_x * math.sin(rotation_angle_radians) + shifted_y * math.cos(rotation_angle_radians)

    final_x = rotated_x + rotation_center_x
    final_y = rotated_y + rotation_center_y

    return round(final_x), round(final_y)


def transform_segment(x_start, y_start, x_end, y_end, shift_x, shift_y, rotation_angle_degrees, rotation_center_x=0, rotation_center_y=0):
    translated_start_x, translated_start_y = translate_point(x_start, y_start, shift_x, shift_y)
    translated_end_x, translated_end_y = translate_point(x_end, y_end, shift_x, shift_y)

    rotated_start_x, rotated_start_y = rotate_point(
        translated_start_x,
        translated_start_y,
        rotation_angle_degrees,
        rotation_center_x,
        rotation_center_y
    )

    rotated_end_x, rotated_end_y = rotate_point(
        translated_end_x,
        translated_end_y,
        rotation_angle_degrees,
        rotation_center_x,
        rotation_center_y
    )

    return rotated_start_x, rotated_start_y, rotated_end_x, rotated_end_y

def draw_raster_line(axis, raster_points, title_text, segment_start, segment_end):
    if not raster_points:
        return

    x_coordinates = [point[0] for point in raster_points]
    y_coordinates = [point[1] for point in raster_points]

    min_x = min(x_coordinates + [segment_start[0], segment_end[0]]) - 2
    max_x = max(x_coordinates + [segment_start[0], segment_end[0]]) + 2
    min_y = min(y_coordinates + [segment_start[1], segment_end[1]]) - 2
    max_y = max(y_coordinates + [segment_start[1], segment_end[1]]) + 2

    for x_coordinate, y_coordinate in raster_points:
        pixel_rectangle = Rectangle(
            (x_coordinate - 0.5, y_coordinate - 0.5),
            1,
            1,
            alpha=0.8
        )
        axis.add_patch(pixel_rectangle)

    axis.plot(
        [segment_start[0], segment_end[0]],
        [segment_start[1], segment_end[1]],
        linestyle='--',
        linewidth=1
    )

    axis.set_title(title_text)
    axis.set_aspect('equal')

    axis.set_xlim(min_x - 0.5, max_x + 0.5)
    axis.set_ylim(min_y - 0.5, max_y + 0.5)

    axis.set_xticks(range(min_x, max_x + 1))
    axis.set_yticks(range(min_y, max_y + 1))

    axis.set_xticks([value - 0.5 for value in range(min_x, max_x + 2)], minor=True)
    axis.set_yticks([value - 0.5 for value in range(min_y, max_y + 2)], minor=True)

    axis.grid(False)
    axis.grid(which='minor')


def main():
    x_start = 2
    y_start = 3
    x_end = 7
    y_end = 9

    shift_x = 4
    shift_y = 2
    rotation_angle_degrees = 10

    transformed_x_start, transformed_y_start, transformed_x_end, transformed_y_end = transform_segment(
        x_start,
        y_start,
        x_end,
        y_end,
        shift_x,
        shift_y,
        rotation_angle_degrees
    )

    natural_original_points = natural_line_algorithm(x_start, y_start, x_end, y_end)
    bresenham_original_points = bresenham_line_algorithm(x_start, y_start, x_end, y_end)

    natural_transformed_points = natural_line_algorithm(
        transformed_x_start,
        transformed_y_start,
        transformed_x_end,
        transformed_y_end
    )

    bresenham_transformed_points = bresenham_line_algorithm(
        transformed_x_start,
        transformed_y_start,
        transformed_x_end,
        transformed_y_end
    )

    print("Исходный отрезок:")
    print(f"A = ({x_start}, {y_start})")
    print(f"B = ({x_end}, {y_end})")
    print()

    print("После переноса и поворота:")
    print(f"A' = ({transformed_x_start}, {transformed_y_start})")
    print(f"B' = ({transformed_x_end}, {transformed_y_end})")
    print()

    print("Точки естественного алгоритма для исходного отрезка:")
    print(natural_original_points)
    print()

    print("Точки алгоритма Брезенхейма для исходного отрезка:")
    print(bresenham_original_points)
    print()

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))

    draw_raster_line(
        axes[0, 0],
        natural_original_points,
        "Исходный отрезок: естественный алгоритм",
        (x_start, y_start),
        (x_end, y_end)
    )

    draw_raster_line(
        axes[0, 1],
        bresenham_original_points,
        "Исходный отрезок: алгоритм Брезенхейма",
        (x_start, y_start),
        (x_end, y_end)
    )

    draw_raster_line(
        axes[1, 0],
        natural_transformed_points,
        "После переноса и поворота: естественный алгоритм",
        (transformed_x_start, transformed_y_start),
        (transformed_x_end, transformed_y_end)
    )

    draw_raster_line(
        axes[1, 1],
        bresenham_transformed_points,
        "После переноса и поворота: алгоритм Брезенхейма",
        (transformed_x_start, transformed_y_start),
        (transformed_x_end, transformed_y_end)
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()