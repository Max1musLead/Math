import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi

def translate(dx, dy):
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])

def rotate(phi):
    return np.array([
        [cos(phi), sin(phi), 0],
        [-sin(phi), cos(phi), 0],
        [0, 0, 1]
    ])

def triangle_center(triangle):
    x = triangle[0, :]
    y = triangle[1, :]

    cx = np.mean(x)
    cy = np.mean(y)

    return cx, cy

def rotate_center(phi, triangle):

    cx, cy = triangle_center(triangle)

    return translate(cx, cy) @ rotate(phi) @ translate(-cx, -cy)

def reflect_x():
    return np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])

def reflect_y():
    return np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

def reflect_line(a, b, c):
    d = a*a + b*b
    return np.array([
        [1-2*a*a/d, -2*a*b/d, -2*a*c/d],
        [-2*a*b/d, 1-2*b*b/d, -2*b*c/d],
        [0, 0, 1]
    ])

def homothety(k):
    return np.array([
        [k, 0, 0],
        [0, k, 0],
        [0, 0, 1]
    ])

def find_midpoint(triangle):
    A = triangle[:2,0]
    B = triangle[:2,1]
    C = triangle[:2,2]

    AB = np.linalg.norm(A-B)
    BC = np.linalg.norm(B-C)
    CA = np.linalg.norm(C-A)

    if AB <= BC and AB <= CA:
        M = (A+B)/2
    elif BC <= CA:
        M = (B+C)/2
    else:
        M = (C+A)/2

    return M

def homothety_point(k, x, y):
    return np.array([
        [k, 0, x * (1 - k)],
        [0, k, y * (1 - k)],
        [0, 0, 1]
    ])

def draw(triangles):
    plt.figure()

    for name, triangle in triangles:
        x = list(triangle[0]) + [triangle[0, 0]]
        y = list(triangle[1]) + [triangle[1, 0]]
        plt.plot(x, y, label=name)

    plt.grid()
    plt.axis("equal")
    plt.legend()
    plt.show()


def main():
    triangle = np.array([
        [1, 4, 2],
        [1, 1, 5],
        [1, 1, 1]
    ])

    dx = 2
    dy = 1

    phi = pi / 2

    a = 1
    b = 1
    c = -1

    k = 2
    m = 2

    T = translate(dx, dy)
    Sx = reflect_x()
    Sy = reflect_y()
    Sl = reflect_line(a, b, c)
    H0 = homothety(k)

    t1 = T @ triangle

    R = rotate_center(phi, t1)
    t2 = R @ t1

    t3 = Sx @ t2
    t4 = Sy @ t3
    t5 = Sl @ t4
    t6 = H0 @ t5

    M = find_midpoint(t6)
    HM = homothety_point(m, M[0], M[1])
    t7 = HM @ t6

    draw([
        ("Original", triangle),
        ("After T", t1),
        ("After R", t2),
        ("After Sx", t3),
        ("After Sy", t4),
        ("After Sl", t5),
        ("After H0", t6),
        ("Final", t7)
    ])

if __name__ == "__main__":
    main()