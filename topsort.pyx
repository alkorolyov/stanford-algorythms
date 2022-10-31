# cython: language_level=3


cdef extern from "Python.h":
    void* PyMem_Calloc(size_t nelem, size_t elsize)

from cpython.mem cimport PyMem_Free
from array_c cimport array_c, create_arr, free_arr, push_back_arr
from stack cimport stack_c, create_stack, push, peek, pop, is_empty_s, free_stack
from graph cimport graph_c, node_c, dict2graph, free_graph, rand_dict_graph
from readg cimport read_graph

from utils import print_func_name
from graphlib import TopologicalSorter, CycleError

cdef void dfs(graph_c* g, size_t s, array_c* top_order, size_t* ft, bint* ft_calculated):
    """    
    DFS using stack with additional finishing time counter for topological sorting.
    :param g: input C graph
    :param s: starting vertex    
    :param top_order: array of vertices sorted topologically, from 0 to max
    :param ft: auxiliary finishing time counter
    :param ft_calculated: auxiliary bint array 
    
    """
    cdef:
        size_t i, j, v
        node_c* nd
        stack_c * stack = create_stack(g.len * 2)
    push(stack, s)

    while not is_empty_s(stack):
        v = peek(stack)
        nd = g.node[v]

        if nd.explored:
            pop(stack)
            if not ft_calculated[v]:
                # print("v", v, "ft:", ft[0])
                ft_calculated[v] = True
                ft[0] += 1
                push_back_arr(top_order, v)
            continue
        else:
            nd.explored = True

        # push each edge of v
        if nd.adj:
            for i in range(nd.adj.size):
                j = nd.adj.items[i]
                if not g.node[j].explored:
                    push(stack, j)

    free_stack(stack)


cdef array_c* topsort(graph_c* g):
    """
    Topological sort for graph
    :param g: C graph
    :return: topologically sorted array of vertices
    """
    cdef:
        size_t i, j
        size_t ft = 0
        bint * ft_calculated = <bint*>PyMem_Calloc(g.len, sizeof(bint))
        array_c* top_order = create_arr(g.len)

    for i in range(g.len):
        if not g.node[i].explored:
            dfs(g, i, top_order, &ft, ft_calculated)
    
    PyMem_Free(ft_calculated)
    return top_order


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

def test_topsort():
    print_func_name()
    graph = {0: [1],
             1: [2],
             2: []}
    cdef:
        size_t i
        graph_c* g = dict2graph(graph)
        array_c * order = topsort(g)

    assert order.items[0] == 2
    assert order.items[1] == 1
    assert order.items[2] == 0

def test_graphlib():
    graph = {0: [1],
             1: [2],
             2: [],
             3: [4],
             4: []}
    ts = TopologicalSorter(graph)
    # print([*ts.static_order()])

    cdef array_c* a = topsort(dict2graph(graph))
    # print_array(a)


def test_topsort_rnd():
    print_func_name()
    DEF n = 50

    cdef:
        size_t i, j
        graph_c* g
        array_c* order

    for i in range(50):
        # m = n * n to assure fully connected graph
        # otherwise topsorting differs from implementation
        graph = rand_dict_graph(n, 2 * n)
        ts = TopologicalSorter(graph)

        try:
            py_order = list(ts.static_order())

            g = dict2graph(graph)
            order = topsort(g)

            for j in range(n):
                assert py_order[j] == order.items[j]

        except CycleError:
            pass

def test_big():
    print_func_name()
    cdef:
        size_t i, j
        graph_c * g = read_graph("scc.txt")
        array_c * order = topsort(g)
    free_graph(g)
    free_arr(order)
