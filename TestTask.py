import numpy as np
import matplotlib.pyplot as plt


def convert_points_to_homogeneous(points):
    ones_column = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack([points, ones_column])
    return homogeneous_points.T


def translation_matrix(delta_x, delta_y):
    return np.array([
        [1, 0, delta_x],
        [0, 1, delta_y],
        [0, 0, 1]
    ])


def rotation_matrix_about_center(angle_radians, center_x, center_y):
    translate_to_origin = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])

    rotation = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians),  np.cos(angle_radians), 0],
        [0, 0, 1]
    ])

    translate_back = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])

    return translate_back @ rotation @ translate_to_origin


def reflection_x_matrix():
    return np.array([
        [1, 0, 0],
        [0,-1, 0],
        [0, 0, 1]
    ])


def reflection_y_matrix():
    return np.array([
        [-1,0,0],
        [0,1,0],
        [0,0,1]
    ])


def homothety_origin_matrix(scale):
    return np.array([
        [scale,0,0],
        [0,scale,0],
        [0,0,1]
    ])


def homothety_about_point_matrix(scale, point_x, point_y):
    translate_to_point = np.array([
        [1,0,-point_x],
        [0,1,-point_y],
        [0,0,1]
    ])

    scale_matrix = np.array([
        [scale,0,0],
        [0,scale,0],
        [0,0,1]
    ])

    translate_back = np.array([
        [1,0,point_x],
        [0,1,point_y],
        [0,0,1]
    ])

    return translate_back @ scale_matrix @ translate_to_point


def find_midpoint_of_shortest_side(points):
    minimum_length = float("inf")
    midpoint = None

    for i in range(3):
        for j in range(i + 1, 3):
            side_length = np.linalg.norm(points[i] - points[j])

            if side_length < minimum_length:
                minimum_length = side_length
                midpoint = (points[i] + points[j]) / 2

    return midpoint


def apply_transformation(transformation_matrix, homogeneous_points):
    transformed_points = transformation_matrix @ homogeneous_points
    return transformed_points[:2].T


def plot_triangles(original_points, transformed_points):
    plt.figure()

    plt.plot(*np.vstack([original_points, original_points[0]]).T, marker='o', label="Original")
    plt.plot(*np.vstack([transformed_points, transformed_points[0]]).T, marker='o', label="Transformed")

    plt.axis('equal')
    plt.grid()
    plt.legend()

    plt.show()


def main():

    points = np.array([
        [1, 1],
        [4, 1],
        [2, 3]
    ])

    homogeneous_points = convert_points_to_homogeneous(points)

    delta_x = 2
    delta_y = 1

    rotation_angle = np.radians(30)

    scale_origin = 1.3
    scale_midpoint = 0.8

    center = points.mean(axis=0)
    center_x = center[0]
    center_y = center[1]

    midpoint = find_midpoint_of_shortest_side(points)
    midpoint_x = midpoint[0]
    midpoint_y = midpoint[1]

    translation = translation_matrix(delta_x, delta_y)
    rotation = rotation_matrix_about_center(rotation_angle, center_x, center_y)
    reflection_x = reflection_x_matrix()
    reflection_y = reflection_y_matrix()
    homothety_origin = homothety_origin_matrix(scale_origin)
    homothety_midpoint = homothety_about_point_matrix(scale_midpoint, midpoint_x, midpoint_y)

    transformation = homothety_midpoint @ homothety_origin @ reflection_x @ reflection_y @ rotation @ translation

    transformed_points = apply_transformation(transformation, homogeneous_points)

    plot_triangles(points, transformed_points)


if __name__ == "__main__":
    main()