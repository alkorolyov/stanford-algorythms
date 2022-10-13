# cython: language_level=3

from stack cimport stack_c, create_stack, push, pop, peek, \
    free_stack, size_s, print_stack, is_empty_s
from graph cimport graph_c, node_c, free_graph, dict2graph, rand_dict_graph
from utils import print_func_name
from libc.stdlib cimport rand


""" ################### Depth-First Search using Recursion ############# """

cdef void dfs_rec(graph_c* g, size_t s, stack_c* output=NULL, size_t* ft=NULL):
    """
    Recursive DFS starting from s vertex
    :param g: C graph
    :param s: starting vertex
    """
    cdef:
        size_t i, j
        node_c* nd

    nd = g.node[s]
    nd.explored = True

    #action
    if output:
        push(output, s)

    for i in range(nd.degree):
        j = nd.adj.items[i]
        if not g.node[j].explored:
            dfs_rec(g, j, output, ft)

    if ft:
        nd.fin_time = ft[0]
        ft[0] += 1

    return


""" ######### Depth-First Search using Stack data-structure ########### """

cdef void dfs_stack(graph_c* g, size_t s, stack_c* output=NULL, size_t* ft=NULL):
    """
    DFS using stack. The difference from classical realization is that during exploration
    we use peek() and vertices stay in the stack. We remove from stack when there is no
    adjacent nodes. Direct analogy to recusrsive procedure and gives us correct finishing
    time values for topological sorting and strongly connected components (SCC).
    :param g: inpur C graph
    :param s: starting vertice
    :param output: (optional) stack for output
    :param ft: (optional) variable for finishing time counter
    :return: void
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
        # v = pop(stack)

        # print("v:", v, g.node[v].explored)
        # print_stack(stack)

        # pop vertex if already explored
        if nd.explored:
            pop(stack)
            if ft and nd.fin_time == -1:
                # print("v", v, "ft:", ft[0])
                nd.fin_time = ft[0]
                ft[0] += 1
            continue
        else:
            nd.explored = True

        # action
        if output:
            push(output, v)
        # print(v)

        # push each edge of v
        for i in range(nd.degree):
            j = nd.adj.items[i]
            if not g.node[j].explored:
                push(stack, j)

    free_stack(stack)


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
    # print_graph(g)
    cdef stack_c* s = create_stack(g.len)
    dfs_rec(g, 0, s)
    # print("dfs len:", size_s(s))
    print_stack(s)
    free_graph(g)
    free_stack(s)

def test_dfs_4():
    print_func_name()
    graph = {0: [1, 2],
             1: [],
             2: [3, 4],
             3: [],
             4: []}
    cdef graph_c* g = dict2graph(graph)
    # print_graph(g)
    cdef stack_c* s = create_stack(g.len)
    dfs_stack(g, 0, s)
    # print("dfs len:", size_s(s))
    # print_stack(s)
    free_graph(g)
    free_stack(s)


def test_dfs_random():
    print_func_name()
    DEF size = 30
    cdef:
        graph_c* g
        node_c * nd
        size_t i, j, k
        stack_c * out = create_stack(size)

    for i in range(1000):
        graph = rand_dict_graph(size, rand() % (size), selfloops=True)
        g = dict2graph(graph)
        dfs_stack(g, 0, out)

        assert size_s(out) <= g.len

        # no duplicates
        for j in range(size_s(out) - 1):
            for k in range(j + 1, size_s(out)):
                assert out.items[j] != out.items[k]

        out.top = -1
        free_graph(g)
    free_stack(out)
