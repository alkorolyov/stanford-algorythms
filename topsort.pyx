# cython: language_level=3


from graph cimport graph_c, dict2graph
from dfs cimport dfs_stack
from utils import print_func_name




cdef void topsort(graph_c* g):
    cdef:
        size_t i
        size_t ft = 0
    for i in range(g.len):
        if not g.node[i].explored:
            dfs_stack(g, i, NULL, &ft)


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

def test_topsort():
    print_func_name()
    graph = {0: [0, 1],
             1: [2],
             2: []}
    cdef:
        size_t i
        graph_c* g = dict2graph(graph)

    topsort(g)

    assert g.node[0].fin_time == 2
    assert g.node[1].fin_time == 1
    assert g.node[2].fin_time == 0

def test_topsort_1():
    print_func_name()
    graph = {0: [1],
             1: [2],
             2: [0]}
    cdef:
        size_t i
        graph_c* g = dict2graph(graph)

    topsort(g)

    for i in range(g.len):
        print("node:", i, "fin time:", g.node[i].fin_time)


