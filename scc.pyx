# cython: language_level=3

# cython: profile=False
# cython: linetrace=False
# cython: binding=False

# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True
# cython: cdivision=True

cimport numpy as cnp
cnp.import_array()

import numpy as np
import pickle
import sys
import random

from stack cimport stack_c, create_stack, push, pop, peek, \
    free_stack, size_s, print_stack, is_empty_s

from libc.stddef cimport ptrdiff_t
from libc.stdlib cimport malloc, realloc, free, EXIT_FAILURE, rand, qsort
from utils import print_func_name
from tqdm import tqdm, trange

def is_empty(f):
    if f.read(1):
        f.seek(0)
        return False
    else:
        return True

def read_file():
    """
    Read dictionary to create adjacency list for original
    and reversed graphs. Vertices named from 0 .. n - 1
    :return: (dict, dict) graph in standardized form
    """
    try:
        with open("scc.pkl", "rb") as f:
            if not is_empty(f):
                print("Reading 'scc.pkl' ... ", end="")
                graph, g_rev = pickle.load(f)
                print("done")
                return graph, g_rev
    except FileNotFoundError:
        pass

    print("Processing 'scc.txt' ... ", end="")

    with open("scc.txt", "r") as f:
        lines = [s for s in f.read().split("\n")[:-1]]

    graph = {}
    g_rev = {}
    max_id = 0

    for l in tqdm(lines):
        # decrease vertex id by 1 to get (0 .. n - 1)
        edge = [(int(s) - 1) for s in l.split(" ")[:-1]]
        max_id = max(edge[0], edge[1], max_id)

        if edge[0] in graph.keys():
            graph[edge[0]].append(edge[1])
        else:
            graph[edge[0]] = [edge[1]]

        if edge[1] in g_rev.keys():
            g_rev[edge[1]].append(edge[0])
        else:
            g_rev[edge[1]] = [edge[0]]

    # fill missing vertices with []
    cdef size_t i
    for i in range(max_id + 1):
        if i not in graph.keys():
            graph[i] = []
        if i not in g_rev.keys():
            g_rev[i] = []

    print("done")

    with open("scc.pkl", "wb") as f:
        pickle.dump((graph, g_rev), f)

    return graph, g_rev


cdef void print_mem(size_t * mem, size_t size):
    cdef size_t i
    for i in range(size):
        addr = hex(<size_t>(&mem[i]))
        val = hex(mem[i])
        print(f"{addr} : {val}")


cdef inline (size_t, size_t) str2int(char* buf):
    """
    Converts space-terminated char string to unsigned integer
    :param buf: pointer to char buffer
    :return: integer value, bytes read including space
    """
    cdef:
        size_t x = 0
        size_t i = 0
    while buf[i] != 0x20:
        x = x * 10 + buf[i] - 48
        i += 1
    return x, i + 1

cdef (size_t, size_t, size_t) read_edge(char* buf):
    cdef:
        size_t i = 0
        size_t v1, v2, rb

    v1, rb = str2int(buf + i)
    i += rb
    v2, rb = str2int(buf + i)
    i += rb
    i += 1
    return v1, v2, i

cdef void read_buff(char* buf, size_t n):
    cdef:
        size_t i = 0
        size_t v1, v2, rb

    while i <= n:
        # read edge
        v1, rb = str2int(buf + i)
        i += rb
        print(v1, i)
        v2, rb = str2int(buf + i)
        i += rb
        print(v2, i)
        i += 1

    # buf = buf + i



""" ################## Linked lists in C ###################### """

ctypedef struct l_list:
    size_t  id
    l_list* next


cdef l_list* create_l(size_t val):
    cdef l_list * l = <l_list *> malloc(sizeof(l_list))
    l.id = val
    l.next = NULL
    return l

cdef void insert_l(l_list* l, size_t val):
    cdef l_list* new_l = <l_list*> malloc(sizeof(l_list))

    # go to the end of l-list
    while l.next:
        l = l.next

    l.next = new_l

    new_l.id = val
    new_l.next = NULL
    return

cdef void print_l(l_list* l):
    cdef l_list* temp = l

    if l == NULL:
        print("[]")
        return

    print("[", end="")
    while temp.next:
        print(temp.id, end=", ")
        temp = temp.next
    # print last element
    print(temp.id, end="]\n")

cdef l_list* arr2list(size_t* arr, size_t n):
    cdef size_t i
    cdef l_list* l = <l_list*> malloc(sizeof(l_list) * n)

    for i in range(n - 1):
        l[i].id = arr[i]
        l[i].next = l + i + 1

    l[n - 1].id = arr[n - 1]
    l[n - 1].next = NULL
    # print_mem(<size_t *>l, 2 * n)
    return l


""" ###################### Queue in C ########################## """


""" Graph structure using adjacency lists (linked lists data structure) """

ctypedef struct node_c:
    # size_t  id
    bint    explored
    size_t  leader
    size_t  finishing_time
    size_t  degree          # total number of connected vertices
    l_list* adj_list        # linked list of connected vertices id

ctypedef struct graph_c:
    size_t      len
    node_c**    node


cdef graph_c* create_graph_c(size_t n):
    """
    Create empty graph of size n
    :param n: size
    :return: pointer C graph
    """
    cdef graph_c* g
    cdef node_c* nd
    cdef size_t i

    g = <graph_c*> malloc(sizeof(graph_c))
    g.len = n
    g.node = <node_c **> malloc(g.len * sizeof(node_c*))
    g.node[0] = <node_c *> malloc(g.len * sizeof(node_c))

    for i in range(n):
        nd = g.node[0] + i
        nd.degree = 0
        nd.explored = False
        nd.finishing_time = -1
        # nd.pushed = False
        nd.adj_list = NULL
        g.node[i] = nd
        # print(f"g.node[{i}]: ", hex(<size_t>g.node[i]))

    # print_mem(<size_t*>g.node, g.len)

    return g

cdef void add_edge(graph_c* g, size_t v1, size_t v2):
    cdef:
        node_c* nd



cdef size_t* read_arr_c(list a):
    """
    Read python list of integers and return C array
    :param a: list Python object
    :return: pointer to C array
    """
    cdef:
        i = 0
        n = len(a)
        size_t * arr = <size_t*>malloc(n * sizeof(size_t))
    for i in range(n):
        arr[i] = a[i]
    return arr


cdef graph_c* read_file_c(str filename):
    with open(filename, "rb") as f:
        py_buf = f.read()

    # x20 - space
    # x0A - new line

    cdef char* c_buf = py_buf
    cdef size_t size = len(py_buf)


cdef graph_c* read_graph_c(dict graph):
    """
    Create C graph from python dict.
    :param graph: graph in standardized form (sorted vertices 0 .. n - 1)
    :return: pointer to C graph
    """
    cdef:
        graph_c* g = create_graph_c(len(graph))
        node_c* nd
        size_t i = 0
        size_t* arr

    for i in range(g.len):
    # for k, v in graph.items():
        nd = g.node[i]
        nd.degree = len(graph[i])
        if graph[i]:
            arr = read_arr_c(graph[i])
            nd.adj_list = arr2list(arr, nd.degree)
            free(arr)
    # print_mem(<size_t*>g.node[0], g.len*5)
    return g

cdef void free_graph(graph_c *g):
    cdef size_t i
    cdef node_c* nd
    for i in range(g.len):
        nd = g.node[i]
        if nd.adj_list:
            free(nd.adj_list)
    free(g.node[0])
    free(g.node)
    free(g)

cdef void print_graph(graph_c *g, size_t length=-1):
    cdef:
        size_t i
        node_c* nd

    for i in range(min(g.len, length)):
        nd = g.node[i]
        print(i, end=": ")
        print_l(nd.adj_list)


cdef void print_g_ext(graph_c *g, size_t length=-1):
    cdef:
        size_t i
        node_c* nd
    for i in range(min(g.len, length)):
        nd = g.node[i]
        print(i, end=": ")
        print_l(nd.adj_list)
        print("   exp:   ", nd.explored)
        print("   ft:    ", hex(nd.finishing_time))
        print("   leader:", nd.leader)



cdef void unexplore_graph(graph_c *g):
    cdef size_t i
    for i in range(g.len):
        g.node[i].explored = False

cdef void mem_size(graph_c *g):
    cdef:
        size_t i
        size_t mem_size = 0
    mem_size += g.len * (sizeof(node_c) + sizeof(node_c*))

    for i in range(g.len):
        mem_size += g.node[i].degree * sizeof(l_list)
    print("c size: ", mem_size)

""" ################### Depth-First Search ############# """

cdef void dfs_rec(graph_c* g, size_t s, stack_c* output=NULL, size_t* ft=NULL):
    """
    Recursive DFS starting from s vertex
    :param g: C graph
    :param s: starting vertex
    """
    cdef size_t i
    cdef l_list* l

    g.node[s].explored = True
    l = g.node[s].adj_list

    #action
    if output:
        push(output, s)

    for i in range(g.node[s].degree):
        if not g.node[l.id].explored:
            dfs_rec(g, l.id, output, ft)
        l = l.next

    if ft:
        g.node[s].finishing_time = ft[0]
        ft[0] += 1

    return

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
        l_list* l
        node_c* nd

    cdef stack_c * stack = create_stack(g.len * 2)
    push(stack, s)

    while not is_empty_s(stack):
        v = peek(stack)
        # v = pop(stack)

        # print("v:", v, g.node[v].explored)
        # print_stack(stack)

        # pop vertex if already explored
        nd = g.node[v]
        if nd.explored:
            pop(stack)
            if ft and nd.finishing_time == -1:
                # print("v", v, "ft:", ft[0])
                nd.finishing_time = ft[0]
                ft[0] += 1

            continue
        else:
            nd.explored = True

        # action
        if output:
            push(output, v)
        nd.leader = s
        # print(v)

        # push each edge of v
        l = g.node[v].adj_list
        for i in range(g.node[v].degree):
            nd = g.node[l.id]
            if not nd.explored:
                push(stack, l.id)
                # print_stack(stack)
            l = l.next

    free_stack(stack)



cdef void dfs_loop_rec(graph_c* g):
    cdef size_t i
    cdef size_t ft = 0

    for i in range(g.len):
        if not g.node[i].explored:
            dfs_rec(g, i, NULL, &ft)

cdef void dfs_loop_1(graph_c* g_rev):
    cdef size_t i
    cdef size_t ft = 0

    for i in range(g_rev.len):
        if not g_rev.node[i].explored:
            dfs_stack(g_rev, i, NULL, &ft)

cdef void dfs_loop_2(graph_c* g, table* ft_table):
    cdef size_t i, j

    for i in range(g.len):
        j = ft_table[i].idx
        if not g.node[j].explored:
            dfs_stack(g, j)


cdef int cmp_table(const void *a, const void *b) nogil:
    cdef:
        table* nd1 = <table*>a
        table* nd2 = <table*>b
    if nd1.val > nd2.val:
        return -1
    elif nd1.val < nd2.val:
        return 1
    else:
        return 0

ctypedef struct table:
    size_t idx
    size_t val

cdef void scc(graph_c* g, graph_c* g_rev, bint debug=False):
    cdef:
        size_t i
        node_c* nd
        table* ft_tab = <table*> malloc(g.len * sizeof(table))

    if debug:
        print_graph(g)
        print("=======")
        print_graph(g_rev)
        print("=== DFS g_rev ====")

    dfs_loop_1(g_rev)

    for i in range(g.len):
        # g.node[i].finishing_time = g_rev.node[i].finishing_time

        ft_tab[i].idx = i
        ft_tab[i].val = g_rev.node[i].finishing_time

        # print(ft_tab[i].idx, ":", ft_tab[i].val)

    qsort(ft_tab, g.len, sizeof(table), cmp_table)

    if debug:
        print_g_ext(g)

    # print("sorted ==== ")
    # for i in range(g.len):
    #     print(ft_tab[i].idx, ":", ft_tab[i].val)

    if debug:
        print("===== DFS loop 2 =====")
    dfs_loop_2(g, ft_tab)

    if debug:
        print_g_ext(g)

    free(ft_tab)

cdef graph_c* reverse_graph(graph_c* g):
    cdef:
        graph_c* r
        size_t i
        size_t j
        node_c* nd
        l_list* l
    r = create_graph_c(g.len)
    for i in range(g.len):
        nd = g.node[i]
        if nd.adj_list:
            l = nd.adj_list
            for j in range(nd.degree):
                # print("i, l.id", i, l.id)
                r_nd = r.node[l.id]
                r_nd.degree += 1
                if r_nd.adj_list:
                    insert_l(r_nd.adj_list, i)
                else:
                    r_nd.adj_list = create_l(i)
                l = l.next
    return r


cdef dict gen_rand_dgraph(size_t n, size_t m, bint selfloops=False):
    cdef:
        size_t i, j, v1, v2

    graph = {}
    for i in range(n):
        graph[i] = []

    for j in range(m):
        v1 = rand() % n
        v2 = rand() % n
        if not selfloops and v1 == v2:
            continue
        graph[v1].append(v2)
    return graph


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

def test_ascii2int():
    print_func_name()
    cdef:
        char * buf = "1234 \n"

    assert str2int(buf)[0] == 1234
    assert str2int(buf)[1] == 5

def test_read_edge_1():
    print_func_name()
    cdef:
        char * buf = "12 34 \n56 78 \n"
        size_t v1, v2, i

    v1, v2, i = read_edge(buf)
    assert v1 == 12
    assert v2 == 34
    # print(v1, v2, i)
    v1, v2, i = read_edge(buf + i)
    assert v1 == 56
    assert v2 == 78
    # print(v1, v2, i)


def test_read_buf_1():
    print_func_name()
    cdef:
        char * buf = "12 34 \n56 78 \n"

    read_buff(buf, 14)



def test_create_graph():
    print_func_name()
    cdef graph_c* g = create_graph_c(2)

    cdef size_t[2] a1 = [2, 2]
    cdef size_t[2] a2 = [1, 1]
    cdef node_c* node = <node_c *> malloc(sizeof(node_c) * 2)

    node[0].adj_list = arr2list(a1, 2)
    node[1].adj_list = arr2list(a2, 2)

    g.node[0] = &node[0]
    g.node[1] = &node[1]

    print_graph(g)
    # free_graph(g)

def test_print_l_list():
    print_func_name()
    cdef l_list first, second, third
    first.id = 1
    second.id = 2
    third.id = 3
    first.next = &second
    second.next = &third
    third.next = NULL
    print_l(&first)

def test_create_l_list():
    print_func_name()
    cdef:
        size_t[3] arr = [1, 2, 3]
        l_list* l = arr2list(&arr[0], 3)
    for i in range(3):
        assert l.id == arr[i]
        l = l.next
    free(l)


def test_insert_l_list():
    print_func_name()
    cdef:
        size_t[3] arr = [1, 2, 3]
        l_list* l = arr2list(&arr[0], 3)
    insert_l(l, 4)
    for i in range(4):
        assert l.id == i + 1
        l = l.next
    free(l)

def test_create_l_list_random():
    DEF size = 1000
    print_func_name()
    cdef size_t[size] arr
    cdef l_list * l
    cdef size_t i, j

    for j in range(100):
        a = np.random.randint(0, 100, size)
        for i in range(size):
            arr[i] = a[i]
        l = arr2list(&arr[0], size)
        for i in range(size):
            assert l.id == a[i]
            l = l.next
        free(l)


def test_read_arr():
    print_func_name()
    cdef size_t* arr = read_arr_c([1, 2, 3])
    assert arr[0] == 1
    assert arr[1] == 2
    assert arr[2] == 3



def test_reverse_graph():
    print_func_name()
    cdef:
        graph_c* g
        graph_c* r
    graph = {0: [1, 2],
             1: [],
             2: []}
    g = read_graph_c(graph)
    r = reverse_graph(g)
    assert r.node[0].adj_list == NULL
    assert r.node[1].adj_list.id == 0
    assert r.node[2].adj_list.id == 0
    free_graph(g)
    free_graph(r)


def test_dfs_1():
    print_func_name()
    graph = {0: [0, 1, 2],
             1: [3],
             2: [],
             3: [4],
             4: []}
    cdef graph_c* g = read_graph_c(graph)
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
    cdef graph_c* g = read_graph_c(graph)
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
    cdef graph_c* g = read_graph_c(graph)
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
    cdef graph_c* g = read_graph_c(graph)
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
        l_list * l
        size_t i, j, k
        stack_c * s = create_stack(size)

    for i in range(1000):
        graph = gen_rand_dgraph(size, rand() % (size), selfloops=True)
        g = read_graph_c(graph)
        dfs_stack(g, 0, s)

        assert size_s(s) <= g.len

        # no duplicates
        for j in range(size_s(s) - 1):
            for k in range(j + 1, size_s(s)):
                assert s.items[j] != s.items[k]

        s.top = -1
        free_graph(g)
    free_stack(s)

def test_dfs_loop_1():
    print_func_name()
    graph = {0: [2, 1],
             1: [],
             2: [3],
             3: [],
             4: []}
    cdef:
        graph_c* g = read_graph_c(graph)
        size_t i

    dfs_loop_rec(g)
    # for i in range(g.len):
    #     print("i:", i, g.node[i].finishing_time)

    assert g.node[0].finishing_time == 3
    assert g.node[1].finishing_time == 2
    assert g.node[2].finishing_time == 1
    assert g.node[3].finishing_time == 0
    assert g.node[4].finishing_time == 4

def test_dfs_loop_2():
    print_func_name()
    graph = {0: [1, 1],
             1: []}
    cdef:
        graph_c* g = read_graph_c(graph)
        size_t i

    dfs_loop_1(g)

    # for i in range(g.len):
    #     print("i:", i, g.node[i].finishing_time)

    assert g.node[0].finishing_time == 1
    assert g.node[1].finishing_time == 0

def test_scc_1():
    print_func_name()
    graph = {0: [],
             1: [],
             2: [0, 1]}
    cdef:
        graph_c* g = read_graph_c(graph)
        graph_c* g_rev = reverse_graph(g)

        size_t i

    scc(g, g_rev, debug=False)
    assert g.node[0].leader == 0
    assert g.node[1].leader == 1
    assert g.node[2].leader == 2


def test_scc_2():
    print_func_name()
    graph = {0: [1, 3],
             1: [0],
             2: [3],
             3: [2]}
    cdef:
        graph_c* g = read_graph_c(graph)
        graph_c * g_rev = reverse_graph(g)
        size_t i

    scc(g, g_rev)
    assert g.node[0].leader == 0
    assert g.node[1].leader == 0
    assert g.node[2].leader == 2
    assert g.node[3].leader == 2

def test_scc_3():
    print_func_name()
    graph = {0: [1, 3],
             1: [0],
             2: [3],
             3: [2, 0]}
    cdef:
        graph_c* g = read_graph_c(graph)
        graph_c * g_rev = reverse_graph(g)
        size_t i

    scc(g, g_rev)
    assert g.node[0].leader == 0
    assert g.node[1].leader == 0
    assert g.node[2].leader == 0
    assert g.node[3].leader == 0


def test_scc_4():
    print_func_name()
    graph = {0: [1],
             1: [0, 2],
             2: [3],
             3: [4],
             4: [2]}
    # graph = {0: [1],
    #          1: [2],
    #          2: [3, 0],
    #          3: [4],
    #          4: [3]}
    cdef:
        graph_c* g = read_graph_c(graph)
        graph_c * g_rev = reverse_graph(g)
        size_t i
        node_c* nd

    scc(g, g_rev)

    l = np.empty(g.len, dtype=np.uint64)
    cdef size_t [:] l_view = l
    for i in range(g.len):
        nd = g.node[i]
        l_view[i] = nd.leader

    print(l)
    val, cnt = np.unique(l, return_counts=True)
    print(np.sort(cnt))

    # print(u)
    # print(np.sort(u, axis=1))



def test_dfs_big():
    print_func_name()
    graph, graph_rev = read_file()
    print("py size:", sys.getsizeof(graph))

    cdef:
        graph_c * g = read_graph_c(graph)
        graph_c * g_rev = read_graph_c(graph_rev)

    mem_size(g)

    cdef stack_c* s = create_stack(g.len * 2)
    print("g.len:", g.len)
    dfs_stack(g, 0, s)
    print("dfs len:", size_s(s))
    free_stack(s)

    free_graph(g)
    free_graph(g_rev)


def test_scc_big():
    print_func_name()
    graph, graph_rev = read_file()

    cdef:
        size_t i
        graph_c * g = read_graph_c(graph)
        graph_c * g_rev = read_graph_c(graph_rev)

    print("Running 'scc()' ... ", end="")
    scc(g, g_rev)
    print("done")
    # print_g_ext(g, 100)

    l = np.empty(g.len, dtype=np.uint64)
    cdef size_t [:] l_view = l
    for i in range(g.len):
        l_view[i] = g.node[i].leader

    val, cnt = np.unique(l, return_counts=True)
    print(val[np.argsort(cnt)][-10:])
    print(np.sort(cnt)[-10:])

    free_graph(g)
    free_graph(g_rev)
