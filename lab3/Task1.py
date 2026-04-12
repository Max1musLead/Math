import heapq

import matplotlib.pyplot as plt

segments = [
    ((0, 0), (5, 5)),      # s1
    ((0, 4), (4, 0)),      # s2

    ((1, 5), (5, 1)),      # s3
    ((0, 3), (5, 4)),      # s4

    ((6, 0), (10, 4)),     # s5
    ((6, 4), (10, 0)),     # s6

    ((0, 6), (4, 8)),      # s7 коллинеарен s8, есть наложение
    ((2, 7), (6, 9)),      # s8

    ((7, 6), (9, 7)),      # s9 коллинеарен s10, наложения нет
    ((11, 8), (13, 9)),    # s10

    ((0, 9), (3, 10)),     # s11
    ((5, 7), (8, 5)),      # s12
    ((9, 1), (13, 2)), # s13
    ((8, 8), (8, 10)),
    ((7, 9), (11, 9)),
    ((10, 8), (10, 10))
]

epsilon = 1e-9

def cross(point1, point2, point3):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

def point_on_segment(point, segment):
    x, y = point
    (x1, y1), (x2, y2) = segment

    return (
        min(x1, x2) - epsilon <= x <= max(x1, x2) + epsilon
        and
        min(y1, y2) - epsilon <= y <= max(y1, y2) + epsilon
    )

def are_collinear(segment1, segment2):
    point1, point2 = segment1
    point3, point4 = segment2

    return (
        abs(cross(point1, point2, point3)) < epsilon
        and
        abs(cross(point1, point2, point4)) < epsilon
    )

def segments_overlap_if_collinear(segment1, segment2):
    if not are_collinear(segment1, segment2):
        return False

    (x1, y1), (x2, y2) = segment1
    (x3, y3), (x4, y4) = segment2

    return not (
        max(x1, x2) < min(x3, x4) - epsilon
        or max(x3, x4) < min(x1, x2) - epsilon
        or max(y1, y2) < min(y3, y4) - epsilon
        or max(y3, y4) < min(y1, y2) - epsilon
    )

def intersection_point(segment1, segment2):
    (x1, y1), (x2, y2) = segment1
    (x3, y3), (x4, y4) = segment2

    a1 = y1 - y2
    b1 = x2 - x1
    c1 = x1 * y2 - x2 * y1

    a2 = y3 - y4
    b2 = x4 - x3
    c2 = x3 * y4 - x4 * y3

    determinant = a1 * b2 - a2 * b1

    if abs(determinant) < epsilon:
        return None

    x = (b1 * c2 - b2 * c1) / determinant
    y = (c1 * a2 - c2 * a1) / determinant

    return x, y

def y_on_segment(segment, x):
    (x1, y1), (x2, y2) = segment

    if abs(x2 - x1) < epsilon:
        return min(y1, y2)

    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def is_vertical(segment):
    (x1, y1), (x2, y2) = segment
    return abs(x2 - x1) < epsilon

# По уравнениям прямых
def find_by_lines(segments_list):
    result = []

    for first_index in range(len(segments_list)):
        for second_index in range(first_index + 1, len(segments_list)):
            first_segment = segments_list[first_index]
            second_segment = segments_list[second_index]

            if are_collinear(first_segment, second_segment):
                continue

            point = intersection_point(first_segment, second_segment)

            if point is None:
                continue

            if point_on_segment(point, first_segment) and point_on_segment(point, second_segment):
                result.append((first_index + 1, second_index + 1, point))

    return result


# Метод косых произведений
def intersect_by_cross_products(segment1, segment2, include_collinear):
    point1, point2 = segment1
    point3, point4 = segment2

    cross1 = cross(point1, point2, point3)
    cross2 = cross(point1, point2, point4)
    cross3 = cross(point3, point4, point1)
    cross4 = cross(point3, point4, point2)

    if (
        abs(cross1) < epsilon
        and abs(cross2) < epsilon
        and abs(cross3) < epsilon
        and abs(cross4) < epsilon
    ):
        if not include_collinear:
            return False
        return segments_overlap_if_collinear(segment1, segment2)

    # Касание концом
    if abs(cross1) < epsilon and point_on_segment(point3, segment1):
        return True
    if abs(cross2) < epsilon and point_on_segment(point4, segment1):
        return True
    if abs(cross3) < epsilon and point_on_segment(point1, segment2):
        return True
    if abs(cross4) < epsilon and point_on_segment(point2, segment2):
        return True

    return (cross1 * cross2 < 0) and (cross3 * cross4 < 0)


def find_by_cross_products(segments_list, include_collinear):
    result = []

    for first_index in range(len(segments_list)):
        for second_index in range(first_index + 1, len(segments_list)):
            if intersect_by_cross_products(
                segments_list[first_index],
                segments_list[second_index],
                include_collinear
            ):
                result.append((first_index + 1, second_index + 1))

    return result


# Заметающая прямая
def normalize_segment(segment):
    point1, point2 = segment

    if point1[0] < point2[0]:
        return point1, point2
    if point1[0] > point2[0]:
        return point2, point1
    if point1[1] <= point2[1]:
        return point1, point2
    return point2, point1


def intersection_info(segment1, segment2, include_collinear):
    if are_collinear(segment1, segment2):
        if not include_collinear:
            return None

        if not segments_overlap_if_collinear(segment1, segment2):
            return None

        left1, right1 = normalize_segment(segment1)
        left2, right2 = normalize_segment(segment2)

        overlap_left_x = max(left1[0], left2[0])
        overlap_right_x = min(right1[0], right2[0])

        if overlap_left_x > overlap_right_x + epsilon:
            return None

        if abs(overlap_left_x - overlap_right_x) < epsilon:
            overlap_point = (overlap_left_x, y_on_segment(segment1, overlap_left_x))
            return {
                "type": "point",
                "point": overlap_point
            }

        overlap_start = (overlap_left_x, y_on_segment(segment1, overlap_left_x))
        overlap_end = (overlap_right_x, y_on_segment(segment1, overlap_right_x))
        return {
            "type": "overlap",
            "start": overlap_start,
            "end": overlap_end
        }

    point = intersection_point(segment1, segment2)

    if point is None:
        return None

    if point_on_segment(point, segment1) and point_on_segment(point, segment2):
        return {
            "type": "point",
            "point": point
        }

    return None


def find_by_sweep_line(segments_list, include_collinear):
    left_event_type = 0
    intersection_event_type = 1
    right_event_type = 2

    event_queue = []
    active_segments = []
    scheduled_intersections = set()
    found_pairs = set()
    event_order = 0

    def add_event(event_x, event_y, event_type, data):
        nonlocal event_order
        heapq.heappush(
            event_queue,
            (event_x, event_y, event_type, event_order, data)
        )
        event_order += 1

    def active_key(segment_index, current_x):
        segment = segments_list[segment_index]
        left_point, right_point = normalize_segment(segment)

        if is_vertical(segment):
            y_value = min(left_point[1], right_point[1])
            slope = float("inf")
            return (y_value, slope, segment_index)

        y_value = y_on_segment(segment, current_x)
        slope = (right_point[1] - left_point[1]) / (right_point[0] - left_point[0])

        return (y_value, slope, segment_index)

    def sort_active(current_x):
        active_segments.sort(key=lambda segment_index: active_key(segment_index, current_x))

    def report_pair(first_index, second_index):
        found_pairs.add(tuple(sorted((first_index + 1, second_index + 1))))

    def schedule_intersection(first_index, second_index, current_x):
        if first_index is None or second_index is None:
            return

        if first_index == second_index:
            return

        info = intersection_info(
            segments_list[first_index],
            segments_list[second_index],
            include_collinear
        )

        if info is None:
            return

        if info["type"] == "overlap":
            report_pair(first_index, second_index)
            return

        point_x, point_y = info["point"]

        if point_x < current_x - epsilon:
            return

        if point_x <= current_x + epsilon:
            report_pair(first_index, second_index)
            return

        event_key = (
            round(point_x, 12),
            round(point_y, 12),
            min(first_index, second_index),
            max(first_index, second_index)
        )

        if event_key not in scheduled_intersections:
            scheduled_intersections.add(event_key)
            add_event(
                point_x,
                point_y,
                intersection_event_type,
                (first_index, second_index)
            )

    for segment_index, segment in enumerate(segments_list):
        left_point, right_point = normalize_segment(segment)
        add_event(left_point[0], left_point[1], left_event_type, segment_index)
        add_event(right_point[0], right_point[1], right_event_type, segment_index)

    while event_queue:
        event_x, event_y, event_type, _, data = heapq.heappop(event_queue)

        if event_type == left_event_type:
            segment_index = data

            if segment_index not in active_segments:
                active_segments.append(segment_index)

            sort_active(event_x + epsilon)
            current_position = active_segments.index(segment_index)

            previous_index = None
            next_index = None

            if current_position > 0:
                previous_index = active_segments[current_position - 1]

            if current_position < len(active_segments) - 1:
                next_index = active_segments[current_position + 1]

            schedule_intersection(segment_index, previous_index, event_x)
            schedule_intersection(segment_index, next_index, event_x)

        elif event_type == right_event_type:
            segment_index = data

            if segment_index not in active_segments:
                continue

            sort_active(event_x - epsilon)
            current_position = active_segments.index(segment_index)

            previous_index = None
            next_index = None

            if current_position > 0:
                previous_index = active_segments[current_position - 1]

            if current_position < len(active_segments) - 1:
                next_index = active_segments[current_position + 1]

            active_segments.remove(segment_index)

            schedule_intersection(previous_index, next_index, event_x)

        else:
            first_index, second_index = data
            report_pair(first_index, second_index)

            if first_index not in active_segments or second_index not in active_segments:
                continue

            sort_active(event_x - epsilon)

            first_position = active_segments.index(first_index)
            second_position = active_segments.index(second_index)

            if first_position > second_position:
                first_position, second_position = second_position, first_position
                first_index, second_index = second_index, first_index

            active_segments[first_position], active_segments[second_position] = (
                active_segments[second_position],
                active_segments[first_position]
            )

            sort_active(event_x + epsilon)

            new_first_position = active_segments.index(first_index)
            new_second_position = active_segments.index(second_index)

            previous_of_first = None
            next_of_first = None
            previous_of_second = None
            next_of_second = None

            if new_first_position > 0:
                previous_of_first = active_segments[new_first_position - 1]
            if new_first_position < len(active_segments) - 1:
                next_of_first = active_segments[new_first_position + 1]

            if new_second_position > 0:
                previous_of_second = active_segments[new_second_position - 1]
            if new_second_position < len(active_segments) - 1:
                next_of_second = active_segments[new_second_position + 1]

            schedule_intersection(previous_of_first, first_index, event_x)
            schedule_intersection(first_index, next_of_first, event_x)
            schedule_intersection(previous_of_second, second_index, event_x)
            schedule_intersection(second_index, next_of_second, event_x)

    return sorted(found_pairs)

def draw_segments(segments_list, intersections_by_lines):
    plt.figure(figsize=(10, 8))

    for index, segment in enumerate(segments_list, start=1):
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], linewidth=2)
        middle_x = (x1 + x2) / 2
        middle_y = (y1 + y2) / 2
        plt.text(middle_x, middle_y, f"s{index}", fontsize=10)

    for first_segment_index, second_segment_index, point in intersections_by_lines:
        x, y = point
        plt.scatter(x, y, s=40)
        plt.text(x + 0.1, y + 0.1, f"({x:.2f}; {y:.2f})", fontsize=9)

    plt.title("Пересечение отрезков")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def main():
    print("Отрезки:")
    for index, segment in enumerate(segments, start=1):
        print(f"s{index} = {segment}")

    intersections_by_lines = find_by_lines(segments)

    intersections_by_cross_non_collinear = find_by_cross_products(
        segments,
        include_collinear=False
    )
    intersections_by_sweep_non_collinear = find_by_sweep_line(
        segments,
        include_collinear=False
    )

    intersections_by_cross_full = find_by_cross_products(
        segments,
        include_collinear=True
    )
    intersections_by_sweep_full = find_by_sweep_line(
        segments,
        include_collinear=True
    )

    print("\nI.a Метод уравнений прямых:")
    for first_segment_index, second_segment_index, point in intersections_by_lines:
        x, y = point
        print(f"s{first_segment_index} и s{second_segment_index} -> ({x:.3f}, {y:.3f})")

    print("\nI.b Метод косых произведений среди неколлинеарных:")
    for first_segment_index, second_segment_index in intersections_by_cross_non_collinear:
        print(f"s{first_segment_index} и s{second_segment_index} пересекаются")

    print("\nI.c Метод заметающей прямой среди неколлинеарных:")
    for first_segment_index, second_segment_index in intersections_by_sweep_non_collinear:
        print(f"s{first_segment_index} и s{second_segment_index} пересекаются")

    print("\nII.b Метод косых произведений с коллинеарными:")
    for first_segment_index, second_segment_index in intersections_by_cross_full:
        print(f"s{first_segment_index} и s{second_segment_index} пересекаются")

    print("\nII.c Метод заметающей прямой с коллинеарными:")
    for first_segment_index, second_segment_index in intersections_by_sweep_full:
        print(f"s{first_segment_index} и s{second_segment_index} пересекаются")

    draw_segments(segments, intersections_by_lines)

if __name__ == "__main__":
    main()