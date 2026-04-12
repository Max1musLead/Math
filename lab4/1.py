"""
Задание 4.1: выпуклая оболочка множества (конспект, с теории выпуклых оболочек).
Оболочка G — алгоритм Грэхема; оболочка D — алгоритм Джарвиса; пересечение,
периметры и площади, точки из G и D строго внутри пересечения.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from first.utils import show_polygons_homogeneous
from third.utils import EPS, orient, point_on_segment

Point = tuple[float, float]


def _dist_sq(a: Point, b: Point) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return dx * dx + dy * dy


def _dedupe_points(points: Iterable[Point]) -> list[Point]:
    seen: set[Point] = set()
    out: list[Point] = []
    for p in points:
        key = (round(p[0], 12), round(p[1], 12))
        if key in seen:
            continue
        seen.add(key)
        out.append((p[0], p[1]))
    return out


def generate_points_disk(
    n: int,
    center: Point,
    radius: float,
    rng: np.random.Generator,
) -> list[Point]:
    """Равномерное распределение в круге (полярные координаты)."""
    pts: list[Point] = []
    for _ in range(n):
        t = 2 * math.pi * rng.random()
        r = radius * math.sqrt(rng.random())
        pts.append((center[0] + r * math.cos(t), center[1] + r * math.sin(t)))
    return pts


def generate_points_G(n: int, rng: np.random.Generator) -> list[Point]:
    return generate_points_disk(n, center=(0.0, 0.0), radius=6.0, rng=rng)


def generate_points_D(k: int, rng: np.random.Generator) -> list[Point]:
    return generate_points_disk(k, center=(4.0, 1.0), radius=5.5, rng=rng)


def _random_convex_quad_in_disk(
    rng: np.random.Generator,
    center: Point,
    radius: float,
) -> list[Point]:
    """Четыре вершины выпуклого четырёхугольника (случайные углы на окружности + лёгкий шум)."""
    cx, cy = center
    base = rng.uniform(0, 2 * math.pi)
    pts: list[Point] = []
    for j in range(4):
        ang = base + j * (math.pi / 2) + rng.uniform(-0.35, 0.35)
        rr = radius * rng.uniform(0.55, 1.0)
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    return pts


def generate_point_sets_with_hull_intersection(
    n: int,
    k: int,
    rng: np.random.Generator,
    *,
    max_attempts: int = 400,
    min_intersection_area: float = 1e-5,
) -> tuple[list[Point], list[Point]]:
    """
    Случайные G и D (каждое — облако в своём круге): conv(G) ∩ conv(D) с ненулевой площадью,
    при этом ни одна из оболочек не содержит другую целиком (у каждой есть вершина снаружи другой).
    Сначала отбор по пересекающимся кругам, затем запасной сценарий с общим quad и крайний — с «ушами».
    """
    assert n >= 4 and k >= 4

    def try_pair() -> tuple[list[Point], list[Point]] | None:
        r1 = rng.uniform(3.5, 9.0)
        r2 = rng.uniform(3.5, 9.0)
        # пересечение кругов непусто, но центры не совпадают — облака «разные»
        d = rng.uniform(0.2 * (r1 + r2), 0.92 * (r1 + r2))
        ang = 2 * math.pi * rng.random()
        c1 = (rng.uniform(-12.0, 12.0), rng.uniform(-10.0, 10.0))
        c2 = (c1[0] + d * math.cos(ang), c1[1] + d * math.sin(ang))
        g_set = generate_points_disk(n, c1, r1, rng)
        d_set = generate_points_disk(k, c2, r2, rng)
        hg = ensure_ccw(graham_hull(g_set))
        hd = ensure_ccw(jarvis_hull(d_set))
        inter = convex_polygon_intersection(hg, hd)
        if len(inter) >= 3 and polygon_area(inter) >= min_intersection_area:
            if _neither_hull_inside_the_other(hg, hd):
                return (g_set, d_set)
        return None

    for _ in range(max_attempts):
        pair = try_pair()
        if pair is not None:
            return pair

    # Гарантия: общий quad + облака по разные стороны; перебор, пока ни одна оболочка не поглощает другую.
    for _ in range(250):
        hub = (rng.uniform(-4.0, 4.0), rng.uniform(-3.0, 3.0))
        quad = _random_convex_quad_in_disk(rng, hub, radius=rng.uniform(1.2, 2.5))
        r1 = rng.uniform(4.0, 8.0)
        r2 = rng.uniform(4.0, 8.0)
        d_sep = rng.uniform(0.35 * (r1 + r2), 0.85 * (r1 + r2))
        phi = 2 * math.pi * rng.random()
        c1 = (
            hub[0] + r1 * 0.4 * math.cos(phi + math.pi),
            hub[1] + r1 * 0.4 * math.sin(phi + math.pi),
        )
        c2 = (hub[0] + d_sep * math.cos(phi), hub[1] + d_sep * math.sin(phi))
        g_set = quad + generate_points_disk(max(0, n - 4), c1, r1, rng)
        d_set = quad + generate_points_disk(max(0, k - 4), c2, r2, rng)
        hg = ensure_ccw(graham_hull(g_set))
        hd = ensure_ccw(jarvis_hull(d_set))
        inter = convex_polygon_intersection(hg, hd)
        if (
            len(inter) >= 3
            and polygon_area(inter) >= min_intersection_area
            and _neither_hull_inside_the_other(hg, hd)
        ):
            return (g_set, d_set)

    # Крайний случай: добавить по «выступающей» точке снаружи чужой оболочки (пересечение сохраняет quad).
    hub = (0.0, 0.0)
    quad = _random_convex_quad_in_disk(rng, hub, radius=2.0)
    r1 = r2 = 6.0
    g_set = quad + generate_points_disk(max(0, n - 4), (-9.0, 0.0), r1, rng)
    d_set = quad + generate_points_disk(max(0, k - 4), (9.0, 0.0), r2, rng)
    hg = ensure_ccw(graham_hull(g_set))
    hd = ensure_ccw(jarvis_hull(d_set))
    if not _neither_hull_inside_the_other(hg, hd):
        cg = (sum(p[0] for p in hg) / len(hg), sum(p[1] for p in hg) / len(hg))
        cd = (sum(p[0] for p in hd) / len(hd), sum(p[1] for p in hd) / len(hd))
        dx, dy = cg[0] - cd[0], cg[1] - cd[1]
        nm = math.hypot(dx, dy) or 1.0
        dx, dy = dx / nm, dy / nm
        g_set = g_set + [(cg[0] + 15.0 * dx, cg[1] + 15.0 * dy)]
        d_set = d_set + [(cd[0] - 15.0 * dx, cd[1] - 15.0 * dy)]
    return (g_set, d_set)


def graham_hull(points: list[Point]) -> list[Point]:
    pts = _dedupe_points(points)
    n = len(pts)
    if n <= 1:
        return pts
    if n == 2:
        return pts

    p0 = min(pts, key=lambda p: (p[1], p[0]))
    rest = [p for p in pts if p != p0]

    def polar_key(p: Point) -> tuple[float, float]:
        ang = math.atan2(p[1] - p0[1], p[0] - p0[0])
        return ang, _dist_sq(p0, p)

    rest.sort(key=polar_key)
    stack: list[Point] = [p0, rest[0]]
    for i in range(1, len(rest)):
        while len(stack) >= 2 and orient(stack[-2], stack[-1], rest[i]) <= EPS:
            stack.pop()
        stack.append(rest[i])
    return stack


def jarvis_hull(points: list[Point]) -> list[Point]:
    pts = _dedupe_points(points)
    n = len(pts)
    if n <= 1:
        return pts
    if n == 2:
        return pts

    start = min(range(n), key=lambda i: (pts[i][1], pts[i][0]))
    hull: list[Point] = []
    p = start
    while True:
        hull.append(pts[p])
        nxt: int | None = None
        for i in range(n):
            if i == p:
                continue
            if nxt is None:
                nxt = i
                continue
            o = orient(pts[p], pts[nxt], pts[i])
            if o < -EPS:
                nxt = i
            elif abs(o) <= EPS and _dist_sq(pts[p], pts[i]) > _dist_sq(pts[p], pts[nxt]):
                nxt = i
        assert nxt is not None
        p = nxt
        if p == start:
            break
    return hull


def polygon_perimeter(poly: list[Point]) -> float:
    if len(poly) < 2:
        return 0.0
    s = 0.0
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        s += math.hypot(b[0] - a[0], b[1] - a[1])
    return s


def polygon_area_signed(poly: list[Point]) -> float:
    """Площадь со знаком (положительна для CCW). Формула шнурка."""
    if len(poly) < 3:
        return 0.0
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a * 0.5


def polygon_area(poly: list[Point]) -> float:
    return abs(polygon_area_signed(poly))


def ensure_ccw(poly: list[Point]) -> list[Point]:
    """Sutherland–Hodgman: внутренняя сторона — слева от рёбер при обходе против часовой стрелки."""
    if len(poly) < 3:
        return poly
    if polygon_area_signed(poly) < -EPS:
        return poly[::-1]
    return poly


def _cross2(x: tuple[float, float], y: tuple[float, float]) -> float:
    return x[0] * y[1] - x[1] * y[0]


def _inside_halfplane(p: Point, a: Point, b: Point) -> bool:
    """Внутренняя полуплоскость слева от ребра a->b (CCW многоугольник)."""
    return orient(a, b, p) >= -EPS


def _line_seg_clip(s: Point, e: Point, a: Point, b: Point) -> Point | None:
    """Пересечение отрезка SE с бесконечной прямой через A, B; возврат, если точка на SE (параметр u на SE)."""
    r = (b[0] - a[0], b[1] - a[1])  # направление прямой AB
    d = (e[0] - s[0], e[1] - s[1])  # направление отрезка SE: P = S + u*d
    den = _cross2(r, d)
    if abs(den) < EPS:
        return None
    sa = (s[0] - a[0], s[1] - a[1])
    u = -_cross2(r, sa) / den
    if -EPS <= u <= 1 + EPS:
        return (s[0] + u * d[0], s[1] + u * d[1])
    return None


def convex_polygon_intersection(poly_a: list[Point], poly_b: list[Point]) -> list[Point]:
    """Пересечение двух выпуклых многоугольников (CCW), Sutherland–Hodgman: subject=A, clip=B."""

    def clip(subject: list[Point], a: Point, b: Point) -> list[Point]:
        if not subject:
            return []
        out: list[Point] = []
        prev = subject[-1]
        prev_in = _inside_halfplane(prev, a, b)
        for curr in subject:
            curr_in = _inside_halfplane(curr, a, b)
            if curr_in:
                if not prev_in:
                    ip = _line_seg_clip(prev, curr, a, b)
                    if ip is not None:
                        out.append(ip)
                out.append(curr)
            elif prev_in:
                ip = _line_seg_clip(prev, curr, a, b)
                if ip is not None:
                    out.append(ip)
            prev = curr
            prev_in = curr_in
        return out

    if len(poly_a) < 3 or len(poly_b) < 3:
        return []

    out = poly_a
    m = len(poly_b)
    for i in range(m):
        a, b = poly_b[i], poly_b[(i + 1) % m]
        out = clip(out, a, b)
        if not out:
            return []

    # Убрать почти-дубликаты подряд
    cleaned: list[Point] = []
    for p in out:
        if cleaned and _dist_sq(p, cleaned[-1]) < EPS * EPS:
            continue
        cleaned.append(p)
    if len(cleaned) >= 2 and _dist_sq(cleaned[0], cleaned[-1]) < EPS * EPS:
        cleaned.pop()
    return cleaned


def point_in_polygon_winding(point: Point, polygon: list[Point]) -> str:
    """INSIDE / OUTSIDE / BOUNDARY (как в third/2.py)."""
    x, y = point
    winding = 0
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        if point_on_segment(point, p1, p2):
            return "BOUNDARY"
        if p1[1] <= y:
            if p2[1] > y and orient(p1, p2, point) > EPS:
                winding += 1
        else:
            if p2[1] <= y and orient(p1, p2, point) < -EPS:
                winding -= 1
    return "INSIDE" if winding != 0 else "OUTSIDE"


def strictly_inside_polygon(p: Point, polygon: list[Point]) -> bool:
    return len(polygon) >= 3 and point_in_polygon_winding(p, polygon) == "INSIDE"


def _hull_has_vertex_strictly_outside(hull: list[Point], container: list[Point]) -> bool:
    """Есть вершина hull строго снаружи container ⇔ conv(hull) не является подмножеством conv(container)."""
    if len(container) < 3 or not hull:
        return True
    return any(point_in_polygon_winding(v, container) == "OUTSIDE" for v in hull)


def _neither_hull_inside_the_other(hg: list[Point], hd: list[Point]) -> bool:
    """Ни conv(G), ни conv(D) не содержит другую оболочку целиком."""
    return _hull_has_vertex_strictly_outside(hg, hd) and _hull_has_vertex_strictly_outside(
        hd, hg
    )


def polygon_to_homogeneous_matrix(polygon: list[Point]) -> np.matrix:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    zs = [1.0] * len(polygon)
    return np.matrix([xs, ys, zs])


def plot_points_and_hulls(
    g_pts: list[Point],
    d_pts: list[Point],
    hull_g: list[Point],
    hull_d: list[Point],
    inter: list[Point],
    inside_g: list[Point],
    inside_d: list[Point],
) -> None:
    """Точки и контуры в одном окне (стиль осей как в third/2.draw_polygon_with_points)."""
    fig, ax = plt.subplots(figsize=(9, 7))

    def ring(poly: list[Point]) -> tuple[list[float], list[float]]:
        if not poly:
            return [], []
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        return xs, ys

    xg, yg = zip(*g_pts) if g_pts else ([], [])
    xd, yd = zip(*d_pts) if d_pts else ([], [])
    ax.scatter(xg, yg, c="tab:blue", s=28, alpha=0.75, label="множество G", zorder=3)
    ax.scatter(xd, yd, c="tab:orange", s=28, alpha=0.75, label="множество D", zorder=3)

    if hull_g:
        xs, ys = ring(hull_g)
        ax.plot(xs, ys, "b-", linewidth=1.8, label=r"$\mathcal{G}$ (Грэхем)")
    if hull_d:
        xs, ys = ring(hull_d)
        ax.plot(xs, ys, "orange", linestyle="--", linewidth=1.8, label=r"$\mathcal{D}$ (Джарвис)")
    if len(inter) >= 2:
        xs, ys = ring(inter)
        ax.plot(xs, ys, "g-.", linewidth=2.0, label=r"$\mathcal{P} = \mathcal{G}\cap\mathcal{D}$")

    if inside_g:
        ix, iy = zip(*inside_g)
        ax.scatter(ix, iy, c="navy", s=55, marker="*", zorder=4, label="G внутри P")
    if inside_d:
        ix, iy = zip(*inside_d)
        ax.scatter(ix, iy, c="darkred", s=55, marker="P", zorder=4, label="D внутри P")

    all_x = [p[0] for p in g_pts + d_pts]
    all_y = [p[1] for p in g_pts + d_pts]
    for poly in (hull_g, hull_d, inter):
        all_x.extend(p[0] for p in poly)
        all_y.extend(p[1] for p in poly)
    if all_x:
        ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
        ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def main() -> None:
    rng = np.random.default_rng(15)
    n, k = 20, 18
    assert n >= 15 and k >= 15

    g_set, d_set = generate_point_sets_with_hull_intersection(n, k, rng)

    hull_g = ensure_ccw(graham_hull(g_set))
    hull_d = ensure_ccw(jarvis_hull(d_set))

    pg, ag = polygon_perimeter(hull_g), polygon_area(hull_g)
    pd, ad = polygon_perimeter(hull_d), polygon_area(hull_d)
    print(f"Оболочка G (Грэхем): периметр = {pg:.6f}, площадь = {ag:.6f}")
    print(f"Оболочка D (Джарвис): периметр = {pd:.6f}, площадь = {ad:.6f}")

    inter = convex_polygon_intersection(hull_g, hull_d)
    if len(inter) < 3:
        print("Пересечение вырождено или пусто (меньше 3 вершин).")
        p_inter, a_inter = 0.0, 0.0
    else:
        p_inter = polygon_perimeter(inter)
        a_inter = polygon_area(inter)
        print(f"Пересечение P: периметр = {p_inter:.6f}, площадь = {a_inter:.6f}")

    inside_g = [p for p in g_set if strictly_inside_polygon(p, inter)]
    inside_d = [p for p in d_set if strictly_inside_polygon(p, inter)]
    print(f"Точки из G строго внутри P ({len(inside_g)}): {inside_g}")
    print(f"Точки из D строго внутри P ({len(inside_d)}): {inside_d}")

    polys: list[tuple[str, np.matrix]] = []
    if len(hull_g) >= 2:
        polys.append(("-", polygon_to_homogeneous_matrix(hull_g)))
    if len(hull_d) >= 2:
        polys.append(("--", polygon_to_homogeneous_matrix(hull_d)))
    if len(inter) >= 2:
        polys.append(("-.", polygon_to_homogeneous_matrix(inter)))
    if polys:
        show_polygons_homogeneous(polys)

    hull_g = ensure_ccw(graham_hull(inside_g + inside_d))

    plot_points_and_hulls(g_set, d_set, hull_g, hull_d, inter, inside_g, inside_d)




if __name__ == "__main__":
    main()
