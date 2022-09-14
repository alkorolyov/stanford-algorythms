import copy
import math

import numpy as np
from random import randrange
from pprint import pprint
from time import time

def read_file() -> dict:
    with open("kargerMinCut.txt", "r") as f:
        # print(f.read().split("\n")[:-1])
        lines = [s for s in f.read().split("\n")[:-1]]
    graph = {}
    for l in lines:
        # vertices = np.array([int(s) for s in l.split("\t")[:-1]], dtype=np.uint16)
        vertices = [int(s) for s in l.split("\t")[:-1]]
        graph[vertices[0]] = vertices[1:]
    return graph


# pprint(read_file(), compact=True)

g = read_file()
sum_deg_v = 0

for v in g:
    # print(v, g[v])
    sum_deg_v += len(g[v])

n = len(g)
m = sum_deg_v // 2
print("n", n)
print("m", m)


g = {1: [2, 3, 4],
     2: [1, 3],
     3: [2, 1, 4],
     4: [1, 3]}

print(g)



def get_size(g: dict) -> int:
    size = 0
    for v in g.values():
        size += len(v)
    return size


def flatten(g: dict):
    return [item for sublist in g.values() for item in sublist]


def random_edge(g: dict, r_idx: int = None):
    """
    Random pair of connected vertices from graph
    :param g: graph dictionary
    :param r_idx: for debug indicate the edge index
    :return: pairs of vertices values
    """
    if r_idx is None:
        r_idx = randrange(get_size(g))

    idx = -1
    for i in g.keys():
        v = g[i]
        idx += len(v)
        if idx >= r_idx:
            delta = idx - r_idx
            return i, v[len(v) - delta - 1]


print(flatten(g))
print(random_edge(g, 8))

# flat_g = flatten(g)
# for i in range(get_size(g)):
#     assert flat_g[i] == random_edge(g, i)[1]

def delete_loops(g: dict, i: int, j: int):
    while True:
        try:
            g[i].pop(g[i].index(j))
            g[j].pop(g[j].index(i))
        except ValueError:
            return


def replace_vertex(g: dict, new: int, old: int):
    for k in g.keys():
        v = g[k]
        for i in range(len(v)):
            if v[i] == old:
                v[i] = new


def transfer_vertices(g: dict, dest: int, source: int):
    for v in g[source]:
        g[dest].append(v)

def delete_vertex(g: dict, idx: int):
    g.pop(idx)

def contract(g: dict):
    i, j = random_edge(g)

    # print("i, j", i, j)

    delete_loops(g, i, j)
    replace_vertex(g, i, j)
    transfer_vertices(g, i, j)
    delete_vertex(g, j)

def mincut(g: dict):
    while len(g) > 2:
        contract(g)
    return len(list(g.values())[0])

def mincut_n(g: dict, n: int):
    min_cut = len(g)
    start_time = time()
    for i in range(n):
        if i % 100 == 0:
            print(f"{i} / {n}: {time() - start_time:.1f}s")
        graph = copy.deepcopy(g)
        m = mincut(graph)
        if m < min_cut:
            min_cut = m
            print("mincut:", min_cut)
    return mincut

graph = {1: [2, 3, 4],
         2: [1, 3],
         3: [2, 1, 4],
         4: [1, 3]}


print("================")
while len(graph) > 2:
    print(graph)
    contract(graph)
print(graph)


print("================")
# # print(g.values())
# print(mincut(g))
# print(g)
# print("================")
#
# g = read_file()
# print(mincut(g))
# print(g)
#
# print("================")
# g = read_file()
# n = len(g)
# N = int(n ** 2 * math.log2(n))
# print(mincut_n(g, N))

