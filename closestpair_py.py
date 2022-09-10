import math
import numpy as np

def distance_py(x1, x2, y1, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def _distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def min_dist_naive_py(P):
    n = P.shape[0]
    d = _distance(P[0], P[1])
    # d = distance_py(P[0, 0], P[1, 0], P[0, 1], P[1, 1])
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = min(_distance(P[i], P[j]), d)
            # d = min(distance_py(P[i, 0], P[j, 0], P[i, 1], P[j, 1]), d)
    return d

def _split_y(mid_x, Py):
    idx_q = Py[:, 0] < mid_x
    idx_r = Py[:, 0] >= mid_x
    return Py[idx_q], Py[idx_r]

def _min_dist_split(Sy, delta):
    for i in range(Sy.shape[0] - 1):
        for j in range((i + 1), min(i + 7, Sy.shape[0])):
            delta = min(delta, _distance(Sy[i], Sy[j]))
    return delta

def _min_dist_py(Px, Py):
    n = Px.shape[0]
    # base cases
    if n == 2:
        return _distance(Px[0], Px[1])
    if n == 3:
        return min(_distance(Px[0], Px[1]),
                   _distance(Px[0], Px[2]),
                   _distance(Px[1], Px[2]))


    # divide
    mid = n // 2
    mid_x = Px[mid, 0]
    Qx, Rx = Px[:mid], Px[mid:]
    Qy, Ry = _split_y(mid_x, Py)

    # conquer
    d1 = _min_dist_py(Qx, Qy)
    d2 = _min_dist_py(Rx, Ry)
    delta = min(d1, d2)

    idx = (Py[:, 0] >= mid_x - delta) & (Py[:, 0] <= mid_x + delta)
    Sy = Py[idx]

    delta = _min_dist_split(Sy, delta)
    return delta

def min_dist_py(P):
    idx = np.argsort(P, axis=0, kind='quicksort')
    Px = P[idx[:, 0]]
    Py = P[idx[:, 1]]
    return _min_dist_py(Px, Py)


