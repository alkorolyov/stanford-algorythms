

cdef extern from "Python.h":
    void* PyMem_Calloc(size_t nelem, size_t elsize)

from cpython.mem cimport PyMem_Free
from libc.stdlib cimport rand, srand
from heap_ex cimport heap_ex, item, create_heap, free_heap, is_empty_h, isin_h, find_h, push_heap, pop_heap, replace_h, print_heap
from array_c cimport array_c, create_arr, create_arr_val, push_back_arr, isin_arr, print_array, free_arr, arr2numpy
from graph cimport graph_c, node_c, create_graph_c, add_edge, dict2graph, free_graph, rand_graph_l, print_graph, print_graph_ext, unexplore_graph
from readg cimport read_graph_l

from dfs cimport dfs_stack

import os
from utils import print_func_name
from time import time
import numpy as np

""" ################# DIJKSTRA's SHORTEST PATH ALGORITHM #################### """


cdef array_c* dijkstra_naive(graph_c* g, size_t s, bint debug=False):
    cdef:
        size_t i, j, score, len, min_score
        size_t w, v, v1, v2
        node_c* nd
        size_t n = g.len
        array_c * exp = create_arr(n)   # array of explored vertices
        array_c* dist = create_arr_val(n, -1)   # array of distances, sorted by vert idx


    # init
    push_back_arr(exp, s)
    g.node[s].explored = True
    dist.items[s] = 0

    # stops when no more edges -> min_score doesn't change
    while min_score != -1:
        min_score = -1  # dijkstra "greedy" score

        # loop through all explored vertices
        for i in range(exp.size):
            v = exp.items[i]
            nd = g.node[v]
            if nd.adj:
                # get all outgoing edges
                for j in range(nd.adj.size):
                    w = nd.adj.items[j]
                    # edge(v, w)
                    # check if w is already explored
                    if not g.node[w].explored:
                        # calculate dijkstra greedy criteria
                        score = dist.items[v] + nd.len.items[j]
                        if score < min_score:
                            min_score = score
                            v1 = v
                            v2 = w
                        if debug:
                            print(f"edge({v}, {w}): {dist.items[v]} + {nd.len.items[j]}")

        # no more edges
        if min_score == -1:
            continue

        # after loop add vertex with min score to explored set
        push_back_arr(exp, v2)
        g.node[v2].explored = True
        dist.items[v2] = min_score

        if debug:
            print(f"edge added: ({v1}, {v2}):", min_score)

    # free
    free_arr(exp)
    return dist


cdef array_c* dijkstra(graph_c* g, size_t s,  bint debug=False):
    """
    Dijkstra's shortest path algorithm using heap.
    Vertex id's (labels) should be mapped to 0 .. n
    :param g: C graph
    :param s: starting vertex
    :return: array of distances
    """
    cdef:
        size_t i, score
        size_t v, w
        item min_h
        node_c* nd
        size_t n = g.len
        array_c* dist = create_arr_val(n, -1)   # array of shortest path distances distances
        heap_ex* h = create_heap(n)

    dist.size = n
    push_heap(h, s, 0)

    while not is_empty_h(h):
        # Explore vertex with min score(distance) from heap
        min_h = pop_heap(h)
        v = min_h.id
        dist.items[v] = min_h.val

        nd = g.node[v]
        nd.explored = True

        if debug:
            print("==== pop, v:", v, "score:", min_h.val)

        # add all outgoing vertices "v - w" to frontier heap
        if nd.adj:
            for i in range(nd.adj.size):
                w = nd.adj.items[i]
                if not g.node[w].explored:
                    score = dist.items[v] + nd.len.items[i]
                    _insert(h, w, score)
            if debug:
                print_heap(h)

    free_heap(h)
    return dist


cdef inline void _insert(heap_ex * h, size_t w, size_t score):
    """
    Insert unexplored vertex to the frontier heap. 
    Check if "w" already added and leave the one with the minimum score (val).

    :param h: frontier heap
    :param w: vertex
    :param score: dijkstra's score
    """
    # check if w already in heap
    # cdef size_t idx = find_h(h, w)
    cdef size_t idx = h.idx[w]

    # if no insert directly
    if idx == -1:
        push_heap(h, w, score)
        return

    # if present: replace with min value
    if score < h.items[idx].val:
        replace_h(h, idx, score)


""" ################################################################ """
""" ########################### Timing ############################# """
""" ################################################################ """

def time_naive():
    print_func_name(end=" ... ")
    DEF n = 500
    cdef:
        graph_c* g
        array_c* d

    srand(2)
    g = rand_graph_l(n, n * n)

    start = time()
    d = dijkstra_naive(g, 0)
    print(f"{time() - start:.3f}s")

    free_arr(d)
    free_graph(g)


def time_heap():
    print_func_name(end=" ... ")
    DEF n = 5000
    cdef:
        graph_c* g
        array_c* d
        size_t i

    srand(2)
    g = rand_graph_l(n, n * n)

    start = time()
    for i in range(1):
        d = dijkstra(g, 0)
        unexplore_graph(g)
    print(f"{time() - start:.3f}s")

    free_arr(d)
    free_graph(g)

""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

TEST_VERTICES = [7,37,59,82,99,115,133,165,188,197]
TEST_PATH = "tests//course2_assignment2Dijkstra//"


def test_naive():
    print_func_name()
    cdef:
        size_t i, j, k
        graph_c* g = create_graph_c(4)

    add_edge(g, 0, 1, 4)
    add_edge(g, 0, 2, 1)
    add_edge(g, 2, 1, 2)
    add_edge(g, 2, 3, 6)
    add_edge(g, 1, 3, 3)

    # print_graph_ext(g)

    cdef array_c *dist = dijkstra_naive(g, 0, debug=False)

    # print_array(dist)
    # print(g.len)
    # print(dist.size)

    assert dist.items[0] == 0
    assert dist.items[1] == 3
    assert dist.items[2] == 1
    assert dist.items[3] == 6

    free_arr(dist)
    free_graph(g)

def test_naive_loops():
    print_func_name()
    cdef:
        size_t i, j, k
        graph_c* g = create_graph_c(4)

    add_edge(g, 0, 1, 4)
    add_edge(g, 1, 0, 4)
    add_edge(g, 0, 2, 1)
    add_edge(g, 2, 0, 1)
    add_edge(g, 2, 1, 2)
    add_edge(g, 1, 2, 2)
    add_edge(g, 2, 3, 6)
    add_edge(g, 3, 2, 6)
    add_edge(g, 1, 3, 3)
    add_edge(g, 3, 1, 3)

    # print_graph_ext(g)

    cdef array_c *dist = dijkstra_naive(g, 0, debug=False)

    # print_array(dist)

    assert dist.items[0] == 0
    assert dist.items[1] == 3
    assert dist.items[2] == 1
    assert dist.items[3] == 6

    free_arr(dist)
    free_graph(g)


def test_naive_self_loops():
    print_func_name()
    print_func_name()
    cdef:
        size_t i, j, k
        graph_c* g = create_graph_c(4)

    add_edge(g, 0, 0, 1)
    add_edge(g, 0, 1, 4)
    add_edge(g, 0, 2, 1)
    add_edge(g, 2, 2, 1)
    add_edge(g, 2, 1, 2)
    add_edge(g, 2, 3, 6)
    add_edge(g, 1, 3, 3)
    add_edge(g, 3, 3, 1)
    # print_graph_ext(g)

    cdef array_c *dist = dijkstra_naive(g, 0, debug=False)
    assert dist.items[0] == 0
    assert dist.items[1] == 3
    assert dist.items[2] == 1
    assert dist.items[3] == 6

    free_arr(dist)
    free_graph(g)

def test_naive_non_conn():
    print_func_name()
    cdef:
        size_t i, j, k
        graph_c* g = create_graph_c(5)

    add_edge(g, 0, 1, 4)
    add_edge(g, 0, 2, 1)
    add_edge(g, 2, 1, 2)
    add_edge(g, 2, 3, 6)
    add_edge(g, 1, 3, 3)
    add_edge(g, 4, 4, 0)


    # print_graph_ext(g)

    cdef array_c *dist = dijkstra_naive(g, 0, debug=False)

    # print_array(dist)
    # print(g.len)
    # print(dist.size)

    assert dist.items[0] == 0
    assert dist.items[1] == 3
    assert dist.items[2] == 1
    assert dist.items[3] == 6
    assert dist.items[4] == -1

    free_arr(dist)
    free_graph(g)

def test_naive_zero_conn():
    print_func_name()
    cdef:
        size_t i, j, k
        graph_c* g = create_graph_c(3)

    add_edge(g, 0, 0, 0)
    add_edge(g, 1, 1, 0)
    add_edge(g, 2, 2, 0)


    # print_graph_ext(g)

    cdef array_c *dist = dijkstra_naive(g, 0, debug=False)

    # print_array(dist)
    # print(g.len)
    # print(dist.size)

    assert dist.items[0] == 0
    assert dist.items[1] == -1
    assert dist.items[2] == -1

    free_arr(dist)
    free_graph(g)

def test_naive_empty():
    print_func_name()
    cdef:
        size_t i, j, k
        graph_c* g = create_graph_c(3)

    # add_edge(g, 0, 0, 0)
    # add_edge(g, 1, 1, 0)
    # add_edge(g, 2, 2, 0)


    # print_graph_ext(g)

    cdef array_c *dist = dijkstra_naive(g, 0, debug=False)

    # print_array(dist)
    # print(g.len)
    # print(dist.size)

    assert dist.items[0] == 0
    assert dist.items[1] == -1
    assert dist.items[2] == -1

    free_arr(dist)
    free_graph(g)

def test_naive_rnd():
    # print_func_name()
    DEF n = 100
    DEF seed = 1
    cdef:
        graph_c* g
        array_c* d

    srand(seed)

    for i in range(100):
        g = rand_graph_l(n, rand() % (n * n), seed)
        # g = rand_graph_l(n, n, seed=1)
        # dijkstra(g, 0)
        d = dijkstra_naive(g, 0)
        free_arr(d)
        free_graph(g)

def test_naive_1():
    print_func_name()
    filename = "input_random_1_4.txt"

    cdef:
        size_t v
        graph_c* g = read_graph_l(TEST_PATH + filename)
        array_c* dist = dijkstra_naive(g, 0, debug=False)

    res = []
    for v in TEST_VERTICES:
        res.append(dist.items[v - 1])

    free_arr(dist)
    free_graph(g)


def _test_single_case_naive(fname="input_random_1_4.txt"):
    input_fname = TEST_PATH + fname

    cdef:
        size_t v
        graph_c* g = read_graph_l(input_fname)
        array_c* dist = dijkstra_naive(g, 0)

    output_fname = TEST_PATH + "output" + fname.split("input")[1]

    res = []
    for v in TEST_VERTICES:
        res.append(dist.items[v - 1])

    with open(output_fname, "r") as f:
        correct = [int(s) for s in f.read().replace("\n", "").split(",")]
        for i in range(len(res)):
            assert correct[i] == res[i]

    free_arr(dist)
    free_graph(g)

def test_single_case_naive():
    print_func_name()
    _test_single_case_naive()

def test_all_cases_naive():
    print_func_name(end=" ... ")
    start = time()
    for f in os.listdir(TEST_PATH):
        if "input" in f:
            _test_single_case_naive(f)

    print(f"{time() - start:.2f}s")

def test_heap():
    print_func_name()
    cdef:
        size_t i, j, k
        graph_c* g = create_graph_c(4)

    add_edge(g, 0, 1, 4)
    add_edge(g, 0, 2, 1)
    add_edge(g, 2, 1, 2)
    add_edge(g, 2, 3, 6)
    add_edge(g, 1, 3, 3)

    # print_graph_ext(g)

    cdef array_c* dist = dijkstra(g, 0, debug=False)
    assert dist.items[0] == 0
    assert dist.items[1] == 3
    assert dist.items[2] == 1
    assert dist.items[3] == 6

    # print_array(dist)

    free_arr(dist)
    free_graph(g)

def test_heap_loops():
    print_func_name()
    cdef:
        size_t i, j, k
        graph_c* g = create_graph_c(4)

    add_edge(g, 0, 1, 4)
    add_edge(g, 1, 0, 4)
    add_edge(g, 0, 2, 1)
    add_edge(g, 2, 0, 1)
    add_edge(g, 2, 1, 2)
    add_edge(g, 1, 2, 2)
    add_edge(g, 2, 3, 6)
    add_edge(g, 3, 2, 6)
    add_edge(g, 1, 3, 3)
    add_edge(g, 3, 1, 3)

    # print_graph_ext(g)

    cdef array_c *dist = dijkstra(g, 0, debug=False)

    # print_array(dist)

    assert dist.items[0] == 0
    assert dist.items[1] == 3
    assert dist.items[2] == 1
    assert dist.items[3] == 6

    free_arr(dist)
    free_graph(g)

def test_heap_self_loops():
    print_func_name()
    cdef:
        size_t i, j, k
        graph_c* g = create_graph_c(4)

    add_edge(g, 0, 0, 1)
    add_edge(g, 0, 1, 4)
    add_edge(g, 0, 2, 1)
    add_edge(g, 2, 2, 1)
    add_edge(g, 2, 1, 2)
    add_edge(g, 2, 3, 6)
    add_edge(g, 1, 3, 3)
    add_edge(g, 3, 3, 1)
    # print_graph_ext(g)

    cdef array_c *dist = dijkstra(g, 0, debug=False)
    assert dist.items[0] == 0
    assert dist.items[1] == 3
    assert dist.items[2] == 1
    assert dist.items[3] == 6

    free_arr(dist)
    free_graph(g)

def test_heap_non_conn():
    print_func_name()
    cdef:
        size_t i, j, k
        graph_c* g = create_graph_c(5)

    add_edge(g, 0, 1, 4)
    add_edge(g, 0, 2, 1)
    add_edge(g, 2, 1, 2)
    add_edge(g, 2, 3, 6)
    add_edge(g, 1, 3, 3)
    add_edge(g, 4, 4, 0)


    # print_graph_ext(g)

    cdef array_c *dist = dijkstra(g, 0, debug=False)

    # print_array(dist)
    # print(g.len)
    # print(dist.size)

    assert dist.items[0] == 0
    assert dist.items[1] == 3
    assert dist.items[2] == 1
    assert dist.items[3] == 6
    assert dist.items[4] == -1

    free_arr(dist)
    free_graph(g)

def test_heap_zero_conn():
    print_func_name()
    cdef:
        size_t i, j, k
        graph_c* g = create_graph_c(3)

    add_edge(g, 0, 0, 0)
    add_edge(g, 1, 1, 0)
    add_edge(g, 2, 2, 0)


    # print_graph_ext(g)

    cdef array_c *dist = dijkstra(g, 0, debug=False)

    # print_array(dist)
    # print(g.len)
    # print(dist.size)

    assert dist.items[0] == 0
    assert dist.items[1] == -1
    assert dist.items[2] == -1

    free_arr(dist)
    free_graph(g)

def test_heap_empty():
    print_func_name()
    cdef:
        size_t i, j, k
        graph_c* g = create_graph_c(3)

    # add_edge(g, 0, 0, 0)
    # add_edge(g, 1, 1, 0)
    # add_edge(g, 2, 2, 0)


    # print_graph_ext(g)

    cdef array_c *dist = dijkstra(g, 0, debug=False)

    # print_array(dist)
    # print(g.len)
    # print(dist.size)

    assert dist.items[0] == 0
    assert dist.items[1] == -1
    assert dist.items[2] == -1

    free_arr(dist)
    free_graph(g)


def test_heap_rnd():
    # print_func_name()
    DEF n = 150
    cdef:
        graph_c* g
        array_c* d

    srand(2)

    for i in range(100):
        g = rand_graph_l(n, rand() % (n * n))
        d = dijkstra(g, 0)
        free_arr(d)
        free_graph(g)


def _test_single_case_heap(fname="input_random_1_4.txt"):
    input_fname = TEST_PATH + fname
    output_fname = TEST_PATH + "output" + fname.split("input")[1]

    cdef:
        size_t v
        graph_c* g = read_graph_l(input_fname)
        array_c* dist = dijkstra(g, 0)

    res = []
    for v in TEST_VERTICES:
        res.append(dist.items[v - 1])

    with open(output_fname, "r") as f:
        correct = [int(s) for s in f.read().replace("\n", "").split(",")]
        for i in range(len(res)):
            assert correct[i] == res[i]

    free_arr(dist)
    free_graph(g)

def test_single_case_heap():
    print_func_name()
    _test_single_case_heap()

def test_all_cases_heap():
    print_func_name(end=" ... ")
    start = time()
    for f in os.listdir(TEST_PATH):
        if "input" in f:
            _test_single_case_heap(f)

    print(f"{time() - start:.2f}s")


