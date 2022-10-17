# cython: language_level=3

from array_c cimport array_c, create_arr, reverse_arr
from graph cimport graph_c, dict2graph
from dfs cimport dfs_stack
from utils import print_func_name


cdef array_c* topsort(graph_c* g):
    """
    Topological sort for graph
    :param g: C graph
    :return: topologically sorted array of vertices
    """
    cdef:
        size_t i, j
        size_t ft = 0
        array_c* ft_order = create_arr(g.len)
    for i in range(g.len):
        if not g.node[i].explored:
            dfs_stack(g, i, NULL, &ft, ft_order)

    reverse_arr(ft_order)

    return ft_order


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

def test_topsort_ft():
    print_func_name()
    graph = {0: [2, 0],
             1: [0],
             2: [1]}
    cdef:
        size_t i
        graph_c* g = dict2graph(graph)
        array_c* ft_order = topsort(g)

    assert ft_order.items[0] == 0
    assert ft_order.items[1] == 2
    assert ft_order.items[2] == 1

    # for i in range(g.len):
    #     print("fin time:", i, "vertex:", ft_order.items[i])
