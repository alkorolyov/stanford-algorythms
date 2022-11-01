

from graph cimport graph_c, node_c, free_graph, dict2graph, rand_dict_graph
from queue_c cimport queue, create_queue, enqueue, dequeue, is_empty_q
from utils import print_func_name, set_stdout, restore_stdout


cdef void bfs(graph_c* g, size_t s):
    cdef:
        size_t i, j, v
        node_c* nd
        queue * q = create_queue(g.len)

    enqueue(q, s)

    while not is_empty_q(q):
        v = dequeue(q)
        nd = g.node[v]

        if nd.explored:
            continue
        else:
            nd.explored = True

        print(v)

        if nd.adj:
            for i in range(nd.adj.size):
                j = nd.adj.items[i]
                if not g.node[j].explored:
                    enqueue(q, j)

""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

def test_bfs():
    print_func_name()
    graph = {0: [1, 2],
             1: [],
             2: [3],
             3: []}
    cdef graph_c* g = dict2graph(graph)

    s = set_stdout()
    bfs(g, 0)
    out = s.getvalue()
    restore_stdout()
    assert out == '0\n1\n2\n3\n'

    free_graph(g)



