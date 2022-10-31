# cython: language_level=3

from time import time
from stack cimport stack_c, create_stack, push, pop, peek, \
    free_stack, size_s, print_stack, is_empty_s
from array_c cimport array_c, push_back_arr, create_arr, free_arr
from readg cimport read_graph, read_graphs
from graph cimport graph_c, node_c, free_graph, dict2graph, rand_dict_graph
from utils import print_func_name
from libc.stdlib cimport rand


""" ################### Depth-First Search using Recursion ############# """

cdef void dfs_rec(graph_c* g, size_t s, array_c* out=NULL):
    """
    Recursive DFS starting from s vertex
    :param g: C graph
    :param s: starting vertex
    :param out: ordered array of visited vertices
    
    """
    cdef:
        size_t i, j
        node_c* nd

    nd = g.node[s]
    nd.explored = True

    #action
    if out:
        push_back_arr(out, s)

    if nd.adj:
        for i in range(nd.adj.size):
            j = nd.adj.items[i]
            if not g.node[j].explored:
                dfs_rec(g, j, out)

    return


""" ######### Depth-First Search using Stack data-structure ########### """

cdef void dfs_stack(graph_c* g, size_t s, array_c* out=NULL):
    """
    DFS using stack. The difference from classical realization is that during exploration
    we use peek() and vertices stay in the stack. We remove from stack when there is no
    adjacent nodes. Direct analogy to recusrsive procedure and gives us correct finishing
    time values for topological sorting and strongly connected components (SCC).
    :param g: inpur C graph
    :param s: starting vertex
    :param out: (optional) array of visited vertices
    """
    cdef:
        size_t i, j, v
        node_c* nd
        stack_c * stack = create_stack(g.len * 2)

    push(stack, s)

    while not is_empty_s(stack):
        v = peek(stack)
        nd = g.node[v]
        nd.leader = s

        # pop vertex if already explored
        if nd.explored:
            pop(stack)
            continue
        else:
            nd.explored = True

        # action
        if out:
            push_back_arr(out, v)

        # push each edge of v
        if nd.adj:
            for i in range(nd.adj.size):
                j = nd.adj.items[i]
                if not g.node[j].explored:
                    push(stack, j)

    free_stack(stack)


cdef void dfs_ordered_loop(graph_c* g, array_c* order):
    cdef size_t i, j
    for i in range(g.len):
        j = order.items[i]
        if not g.node[j].explored:
            dfs_stack(g, j)


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """


def test_dfs_1():
    print_func_name()
    graph = {0: [0, 1, 2],
             1: [3],
             2: [],
             3: [4],
             4: []}
    cdef graph_c* g = dict2graph(graph)
    # print_graph(g)
    dfs_rec(g, 0)
    free_graph(g)

def test_dfs_2():
    print_func_name()
    graph = {0: [0, 1, 2],
             1: [3],
             2: [],
             3: [4],
             4: []}
    cdef graph_c* g = dict2graph(graph)
    # print_graph(g)
    dfs_stack(g, 0)
    free_graph(g)

def test_dfs_3():
    print_func_name()
    graph = {0: [2, 1, 4],
             1: [],
             2: [4, 3],
             3: [],
             4: []}
    cdef graph_c* g = dict2graph(graph)
    cdef array_c* out = create_arr(g.len)

    dfs_rec(g, 0, out)

    assert out.items[0] == 0
    assert out.items[1] == 2
    assert out.items[2] == 4
    assert out.items[3] == 3
    assert out.items[4] == 1
    # print_stack(out)

    free_graph(g)
    free_arr(out)

def test_dfs_4():
    print_func_name()
    graph = {0: [1, 2],
             1: [],
             2: [3, 4],
             3: [],
             4: []}
    cdef graph_c* g = dict2graph(graph)
    # print_graph(g)
    cdef array_c* out = create_arr(g.len)
    dfs_stack(g, 0, out)
    # print("dfs len:", size_s(out))

    assert out.items[0] == 0
    assert out.items[1] == 2
    assert out.items[2] == 4
    assert out.items[3] == 3
    assert out.items[4] == 1

    # print_stack(out)
    free_graph(g)
    free_arr(out)

def test_dfs_random():
    print_func_name()
    DEF size = 30
    cdef:
        graph_c* g
        node_c * nd
        size_t i, j, k
        array_c * out = create_arr(size)

    for i in range(1000):
        graph = rand_dict_graph(size, rand() % (size), selfloops=True)
        g = dict2graph(graph)
        dfs_stack(g, 0, out)

        assert out.size <= g.len

        # no duplicates
        for j in range(out.size - 1):
            for k in range(j + 1, out.size):
                assert out.items[j] != out.items[k]

        out.size = 0
        free_graph(g)
    free_arr(out)

def test_dfs_big():
    print_func_name(end=" ... ")
    cdef:
        graph_c* g
        graph_c* g_rev

    start = time()
    g = read_graph("scc.txt")
    dfs_stack(g, 0)
    print(f"{time() - start:.2f}s")

def test_dfs_loop_big():
    print_func_name(end=" ... ")
    cdef:
        size_t i
        graph_c* g
        graph_c* g_rev

    g = read_graph("scc.txt")

    cdef:
        array_c * order = create_arr(g.len)
    for i in range(g.len):
        order.items[i] = i
    order.size = g.len

    start = time()
    dfs_ordered_loop(g, order)
    print(f"{time() - start:.2f}s")


