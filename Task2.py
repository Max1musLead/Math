import numpy as np
import matplotlib.pyplot as plt
from math import tan, pi

def translate(dx, dy):
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])

def homothety(k):
    return np.array([
        [k, 0, 0],
        [0, k, 0],
        [0, 0, 1]
    ])

def homothety_point(k, x, y):
    return np.array([
        [k, 0, x * (1 - k)],
        [0, k, y * (1 - k)],
        [0, 0, 1]
    ])

def shear_y(k):
    return np.array([
        [1, 0, 0],
        [k, 1, 0],
        [0, 0, 1]
    ])

def draw(polygons):

    plt.figure()

    for name, poly in polygons:
        x = list(poly[0]) + [poly[0, 0]]
        y = list(poly[1]) + [poly[1, 0]]
        plt.plot(x, y, label = name)

    plt.grid()
    plt.axis("equal")
    plt.legend()
    plt.show()

def main():
    K = (1, 3)
    L = (3, 3)
    M = (3, 1)
    N = (1, 1)

    square = np.array([
        [K[0], L[0], M[0], N[0]],
        [K[1], L[1], M[1], N[1]],
        [1, 1, 1, 1]
    ])

    r = (-3 * (K[0] - M[0]), -3 * (K[1] - M[1]))
    A = (K[0] + r[0], K[1] + r[1])

    T = translate(*r)
    sq1 = T @ square

    H = homothety_point(2, *A)
    sq2 = H @ sq1

    sh = -tan(pi / 3)
    Sh = shear_y(sh)
    sq3 = Sh @ sq2

    T2 = translate(0, -sh * A[0])
    sq4 = T2 @ sq3

    F = T2 @ Sh @ H @ T

    abcd = F @ square

    F_inv = np.linalg.inv(F)

    square_back = F_inv @ abcd

    draw([
        ("Square KLMN", square),
        ("After T", sq1),
        ("After H", sq2),
        ("After Shear", sq3),
        ("Parallelogram ABCD", sq4),
    ])

    draw([
        ("Square KLMN", square),
        ("ABCD", abcd),
    ])

    draw([
        ("ABCD F^-1", square_back),
        ("ABCD", abcd),
    ])


if __name__ == "__main__":
    main()