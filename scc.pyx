# cython: language_level=3

# cython: profile=True
# cython: linetrace=True
# cython: binding=True

# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True
# cython: cdivision=False

cimport numpy as cnp

from libc.stdlib cimport malloc, free
from utils import print_func_name
from tqdm import tqdm
import pickle

def is_empty(f):
    if f.read(1):
        f.seek(0)
        return False
    else:
        return True

def read_file() -> dict:
    try:
        with open("scc.pkl", "rb") as f:
            if not is_empty(f):
                print("Reading 'scc.pkl' ... ")
                graph = pickle.load(f)
                return graph
    except FileNotFoundError:
        pass

    print("Processing 'scc.txt' ... ")

    with open("scc.txt", "r") as f:
        lines = [s for s in f.read().split("\n")[:-1]]

    v_idx = 1
    graph = {1 : []}

    for l in tqdm(lines):
        edge = [int(s) for s in l.split(" ")[:-1]]
        if edge[0] == v_idx:
            graph[edge[0]].append(edge[1])
        else:
            v_idx = edge[0]
            graph[v_idx] = [edge[1]]

    with open("scc.pkl", "wb") as f:
        pickle.dump(graph, f)

    return graph


cdef void print_mem(size_t * mem, size_t size):
    cdef size_t i
    for i in range(size):
        addr = hex(<size_t>(&mem[i]))
        val = hex(mem[i])
        print(f"{addr} : {val}")

""" ################## Linked lists in C ###################### """

ctypedef struct l_list:
    size_t  id
    l_list* next

cdef l_list* create_l(size_t* arr, size_t n):
    cdef size_t i
    cdef l_list* l = <l_list*> malloc(sizeof(l_list) * n)

    for i in range(n - 1):
        l[i].id = arr[i]
        l[i].next = l + i + 1

    l[n - 1].id = arr[n - 1]
    l[n - 1].next = NULL

    # print_mem(<size_t *>l, 2 * n)

    return l

cdef void insert_l(l_list* l):
    return

cdef void print_l(l_list* l):
    cdef l_list* temp = l
    print("[", end="")
    while temp.next:
        print(temp.id, end=", ")
        temp = temp.next
    # print last element
    print(temp.id, end="]\n")

""" ###################### Stack in C ########################## """



""" ###################### Queue in C ########################## """


""" Graph structure using adjacency lists (linked lists data structure) """

ctypedef struct node_c:
    size_t  id
    bint    explored
    size_t  leader
    size_t  finishing_time
    size_t  degree          # total number of connected vertices
    l_list* adj_list        # linked list of connected vertices id

ctypedef struct graph_c:
    size_t      len
    node_c**    node


cdef graph_c* create_graph_c(size_t n):
    cdef graph_c* g
    cdef size_t i

    g = <graph_c*> malloc(sizeof(graph_c))
    g.len = n
    g.node = <node_c **> malloc(g.len * sizeof(node_c*))
    for i in range(n):
        g.node[i] = NULL

    # print_debug(g)

    return g

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

cdef graph_c* read_graph_c(dict graph):
    cdef:
        graph_c* g = create_graph_c(len(graph))
        node_c* node
        size_t i = 0
        size_t* arr
    for k, v in graph.items():
        if v:
            node = <node_c*>malloc(sizeof(node_c))
            g.node[i] = node
            g.node[i].id = k
            g.node[i].degree = len(v)
            arr = read_arr_c(v)
            g.node[i].adj_list = create_l(arr, len(v))
            free(arr)
            i += 1
    return g

cdef void free_graph(graph_c *g):
    cdef size_t i
    for i in range(g.len):
        if g.node[i]:
            free(g.node[i].adj_list)
            free(g.node[i])
    free(g.node)
    free(g)

cdef void print_graph(graph_c *g):
    cdef size_t i
    cdef node_c* temp
    for i in range(g.len):
        temp = g.node[i]
        if temp:
            print(temp.id, end=": ")
            print_l(temp.adj_list)


""" ######################### UNIT TESTS ########################### """


def test_create_graph():
    print_func_name()
    cdef graph_c* g = create_graph_c(2)
    cdef size_t[2] a1 = [2, 2]
    cdef size_t[2] a2 = [1, 1]
    cdef node_c* node = <node_c *> malloc(sizeof(node_c) * 2)

    node[0].id = 1
    node[0].adj_list = create_l(a1, 2)
    node[1].id = 2
    node[1].adj_list = create_l(a2, 2)

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
        l_list* l = create_l(&arr[0], 3)
        l_list* temp = l
        size_t i = 1
    while temp.next:
        assert temp.id == i
        temp = temp.next
        i += 1
    # print_l(l)
    free(l)

def test_read_arr():
    print_func_name()
    cdef size_t* arr = read_arr_c([1, 2, 3])
    assert arr[0] == 1
    assert arr[1] == 2
    assert arr[2] == 3

def test_read_graph():
    print_func_name()
    graph = {1: [1, 2],
             2: [],
             3: [2]}
    cdef graph_c* g = read_graph_c(graph)
    assert g.node[0].id == 1
    assert g.node[1].id == 3

    cdef l_list* temp = g.node[0].adj_list
    assert temp.id == 1
    temp = temp.next
    assert temp.id == 2
    temp = temp.next
    assert temp == NULL

    temp = g.node[1].adj_list
    assert temp.id == 2
    temp = temp.next
    assert temp == NULL

    free_graph(g)
    # print(graph)
    # print_graph(g)

def test_read_graph_1(graph: dict):
    print_func_name()
    cdef graph_c* g = read_graph_c(graph)
    free_graph(g)



