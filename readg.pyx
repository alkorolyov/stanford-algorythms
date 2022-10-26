# cython: language_level=3

from time import time
from utils import print_func_name, set_stdout, restore_stdout
from graph cimport graph_c, create_graph_c, add_edge, print_graph, print_graph_ext, free_graph
from array_c cimport array_c, create_arr, resize_arr, free_arr, print_array, max_arr

""" 
    ######################## Directed graphs ######################### 
    File format:
    Space separated list of edges (v1, v2) per each line. 
    [v1] [space] [v2] ([space], [caret ret]) [new line]
    or
    [v1] [0x20] [v2] ([0x20], [0x0D]) [0x0A]
    
    Possibly missing indices. Need to loop through all possible vertices
    to find size of the graph.
"""

cdef inline (size_t, size_t) str2int(char* buf):
    """
    Converts (tab/space/newline) terminated string to unsigned integer
    Terminating symbols:
    0x20 - space
    0x09 - tab 
    0x0D - carret return \r
    0x0A - new line \n
    
    :param buf: pointer to char buffer
    :return: integer value, bytes read including terminating symbol
    """
    cdef:
        size_t x = 0
        size_t i = 0
        size_t s

    while buf[i] != 0x20 and buf[i] != 0x0D and buf[i] != 0x0A and buf[i] != 0x09:
        # print(hex(buf[i]))
        x = x * 10 + buf[i] - 0x30
        i += 1
    return x, i + 1

cdef inline (size_t, size_t, size_t) read_edge(char* buf):
    """
    Read edge from space separated line. Terminated by \n or \r\n
    :param buf: input char buf
    :return: [v1, v2] indexes, total bytes read
    """
    cdef:
        size_t i = 0
        size_t v1, v2, rb

    v1, rb = str2int(buf + i)
    i += rb
    v2, rb = str2int(buf + i)
    i += rb

    # in case \r\n
    if buf[i] == 0x0D:
        i += 1

    if buf[i] == 0x0A:
        i += 1

    return v1, v2, i

cdef void read_buff(char* buf, size_t n):
    cdef:
        size_t i = 0
        size_t v1, v2, rb

    while i < n:
        # read edge
        v1, rb = str2int(buf + i)
        i += rb
        print(v1, i)
        v2, rb = str2int(buf + i)
        i += rb
        print(v2, i)

        # in case \r\n
        if buf[i] == 0x0D:
            i += 1

        if buf[i] == 0x0A:
            i += 1

cdef array_c* read_array(str filename):
    """
    Read graph as C array of directed edges.
    File Format: v1 [space] v2 ([space], [caret return]) [new line] ...
    :param filename: str
    :return: array of edges [v1, v2]
    """

    with open(filename, 'rb') as f:
        read_buf = f.read()
    cdef:
        size_t v1, v2, rb
        size_t i = 0
        size_t j = 0
        char * ptr = read_buf
        size_t length = len(read_buf)
        array_c * v_arr = create_arr(4)

    while i < length:
        v1, v2, rb = read_edge(ptr + i)

        # normalize vertices
        v1 -= 1
        v2 -= 1

        i += rb

        if 2*j >= v_arr.capacity:
            resize_arr(v_arr)

        v_arr.items[2*j] = v1
        v_arr.items[2*j + 1] = v2
        j += 1

    v_arr.size = 2*j
    return v_arr


cdef graph_c* arr2graph(array_c* arr):
    cdef:
        graph_c* g
        size_t i
        size_t v1, v2, g_len

    g_len = max_arr(arr) + 1
    g = create_graph_c(g_len)

    for i in range(arr.size//2):
        v1 = arr.items[2*i]
        v2 = arr.items[2*i + 1]
        add_edge(g, v1, v2)
    return g

cdef (graph_c*, graph_c*) arr2graphs(array_c* arr):
    cdef:
        graph_c* g
        graph_c* g_rev
        size_t i
        size_t v1, v2, g_len

    g_len = max_arr(arr) + 1
    g = create_graph_c(g_len)
    g_rev = create_graph_c(g_len)

    for i in range(arr.size//2):
        v1 = arr.items[2*i]
        v2 = arr.items[2*i + 1]
        add_edge(g, v1, v2)
        add_edge(g_rev, v2, v1)
    return g, g_rev


cdef graph_c* read_graph(str filename):
    cdef:
        array_c* arr
        graph_c* g

    arr = read_array(filename)

    # print("g_len: ", g_len)
    # print_array(arr)
    g = arr2graph(arr)
    free_arr(arr)
    return g

cdef (graph_c*, graph_c*) read_graphs(str filename):
    cdef:
        array_c* arr
        graph_c* g
        graph_c* g_rev

    arr = read_array(filename)
    # print("g_len: ", g_len)
    # print_array(arr)
    g, g_rev = arr2graphs(arr)
    free_arr(arr)
    return g, g_rev

""" 
    ######################## Directed graphs with lengths ######################### 
    File format:
    Tab separated adjacency list with edge lengths.     
    [i] [tab] [v1] [coma] [length] [tab] ... ([caret ret]) [new line]
    or
    [i] [0x09] [v1] [0x2C] [length] [0x09] ... [0x0D] [0x0A]
    
    No missing vertices [i]. Total nodes - last index [i]
"""

cdef graph_c* read_graph_l(str filename):

    with open(filename, "r") as f:
        read_buf = f.readlines()

    last_line = read_buf[-1]
    graph_size = int(last_line.split("\t")[0])

    cdef graph_c* g = create_graph_c(graph_size)

    for line in read_buf:
        line = line.replace("\n", "").replace("\r", "")
        line = line.split("\t")
        # print(line)
        v = int(line[0])
        edges = line[1:]
        # print("v:", v)
        # print("edges")
        for e in edges:
            e = [int(s) for s in e.split(",")]
            add_edge(g, v - 1, e[0] - 1, e[1])
            # print(e)

    return g


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """


def test_ascii2int():
    print_func_name()
    cdef:
        char * buf = "1234 \r\n" # 0x20 0x0D 0x0A

    assert str2int(buf)[0] == 1234
    assert str2int(buf)[1] == 5

def test_ascii2int_1():
    print_func_name()
    cdef:
        char * buf = "1234 \n" # 0x20 0xA0

    assert str2int(buf)[0] == 1234
    assert str2int(buf)[1] == 5

def test_ascii2int_2():
    print_func_name()
    cdef:
        char * buf = "1234\n" # 0xA0

    assert str2int(buf)[0] == 1234
    assert str2int(buf)[1] == 5


def test_read_edge_spc_n():
    print_func_name()
    cdef:
        char * buf = "12 34 \n5"
        size_t v1, v2, i

    v1, v2, i = read_edge(buf)
    assert v1 == 12
    assert v2 == 34
    assert i == 7
    assert buf[i] == 0x35

def test_read_edge_spc_rn():
    print_func_name()
    cdef:
        char * buf = "12 34 \r\n5"
        size_t v1, v2, i

    v1, v2, i = read_edge(buf)
    assert v1 == 12
    assert v2 == 34
    assert i == 8
    assert buf[i] == 0x35

    # v1, v2, i = read_edge(buf + i)
    # assert v1 == 56
    # assert v2 == 78
    # assert i == 8
    # v1, v2, i = read_edge(buf + i)
    # assert v1 == 1
    # assert v2 == 2

def test_read_edge_rn():
    print_func_name()
    cdef:
        char * buf = "12 34\r\n5"
        size_t v1, v2, i

    v1, v2, i = read_edge(buf)
    assert v1 == 12
    assert v2 == 34
    assert i == 7
    assert buf[i] == 0x35


def test_read_edge_n():
    print_func_name()
    cdef:
        char * buf = "12 34\n5"
        size_t v1, v2, i

    v1, v2, i = read_edge(buf)
    assert v1 == 12
    assert v2 == 34
    assert i == 6
    assert buf[i] == 0x35



def test_read_buf_1():
    print_func_name()
    cdef:
        char * buf = "12 34 \n567 8 \r\n"

    s = set_stdout()
    read_buff(buf, 15)
    out = s.getvalue()
    restore_stdout()

    assert out == '12 3\n34 6\n567 11\n8 13\n'


def test_read_array():
    print_func_name()
    cdef:
        array_c* arr
    arr = read_array("scc_small.txt")
    assert arr.items[0] == 0
    assert arr.items[1] == 0
    assert arr.items[2] == 0
    assert arr.items[3] == 1
    assert arr.items[4] == 1
    assert arr.items[5] == 0
    assert arr.items[6] == 1
    assert arr.items[7] == 2
    assert arr.size == 8
    free_arr(arr)

def test_read_graph():
    print_func_name()
    cdef graph_c* g
    g = read_graph("scc_small.txt")
    assert g.node[0].adj.items[0] == 0
    assert g.node[0].adj.items[1] == 1
    assert g.node[1].adj.items[0] == 0
    assert g.node[1].adj.items[1] == 2
    free_graph(g)
    # print_graph(g)


def test_read_big():
    print_func_name(end=" ... ")

    start_time = time()

    cdef graph_c* g = read_graph("scc.txt")

    print(f"{time() - start_time:.2f}s")

    free_graph(g)

def test_read_big_pair():
    print_func_name(end="\t")

    cdef:
        graph_c* g
        graph_c* g_rev

    start_time = time()

    g, g_rev = read_graphs("scc.txt")

    print(f"{time() - start_time:.2f}s")

    free_graph(g)

def test_read_graph_l():
    print_func_name()
    path = "tests//course2_assignment2Dijkstra//"
    filename = "input_random_1_4.txt"

    cdef:
        graph_c* g = read_graph_l(path + filename)

    # print_graph_ext(g, 5)
    # print_array(g.node[3].adj)
    free_graph(g)
