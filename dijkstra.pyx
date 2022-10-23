# cython: language_level=3

from heap_c cimport heap_c, create_heap, insert_h, extract_min
from array_c cimport array_c, create_arr, create_arr_val, push_back_arr, isin_arr, print_array, free_arr, arr2numpy
from graph cimport graph_c, node_c, create_graph_c, add_edge, dict2graph, free_graph, print_graph, print_graph_ext
from readg cimport read_graph_l

import os
from utils import print_func_name
import numpy as np


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
    dist.items[s] = 0

    while exp.size < n:
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
                    if not isin_arr(exp, w):
                        # calculate dijkstra greedy criteria
                        score = dist.items[v] + nd.len.items[j]
                        if score < min_score:
                            min_score = score
                            v1 = v
                            v2 = w
                        if debug:
                            print(f"edge({v}, {w}): {dist.items[v]} + {nd.len.items[j]}")


        # after loop add vertex with min score to explored set
        push_back_arr(exp, v2)
        dist.items[v2] = min_score

        if debug:
            print(f"edge added: ({v1}, {v2}):", min_score)
    return dist


cdef array_c* dijkstra(graph_c* g, size_t s):
    cdef:
        size_t i, j, score, len, min_score
        size_t w, v, v1, v2
        node_c* nd
        size_t n = g.len
        array_c * exp = create_arr(n)   # array of explored vertices
        array_c* dist = create_arr_val(n, -1)   # array of distances, sorted by vert idx



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

def _test_single_case_naive(fname="input_random_1_4.txt"):    
    input_fname = TEST_PATH + fname
    output_fname = TEST_PATH + "output" + fname.split("input")[1]

    cdef:
        size_t v
        graph_c* g = read_graph_l(input_fname)
        array_c* dist = dijkstra_naive(g, 0)
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

def test_all_casses_naive():
    print_func_name(end=" ... ")
    i = 0
    for f in os.listdir(TEST_PATH):
        if "input" in f:
            _test_single_case_naive(f)
            i += 1

    print(f"{i} passed")




