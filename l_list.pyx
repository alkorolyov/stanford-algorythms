
from libc.stdlib cimport malloc, realloc, free
from c_utils cimport max_st


""" ################## Linked lists in C ###################### """

DEF MIN_SIZE = 10

ctypedef struct l_item:
    size_t value
    size_t* next

ctypedef struct list_c:
    size_t capacity
    size_t size
    l_item* items

cdef list_c* make_list(size_t n):
    cdef:
        list_c* l = <list_c *> malloc(sizeof(list_c))
    l.capacity = max_st(MIN_SIZE, n)
    l.items = <l_item*> malloc(sizeof(l_item) * l.capacity)


cdef void resize_list(list_c* l):
    l.capacity *= 2
    l.items = <l_item*> realloc(l.items, l.capacity * sizeof(l_item))


cdef void insert_list(list_c* l, size_t x):
    if l.size == l.capacity:
        resize_list(l)



cdef list_c* create_l(size_t val):
    cdef list_c * l = <list_c *> malloc(sizeof(list_c))
    l.id = val
    l.next = NULL
    return l

cdef void insert_l(list_c* l, size_t val):
    cdef list_c* new_l = <list_c*> malloc(sizeof(list_c))

    # go to the end of l-list
    while l.next:
        l = l.next

    l.next = new_l

    new_l.id = val
    new_l.next = NULL
    return

cdef void print_l(list_c* l):
    cdef list_c* temp = l

    if l == NULL:
        print("[]")
        return

    print("[", end="")
    while temp.next:
        print(temp.id, end=", ")
        temp = temp.next
    # print last element
    print(temp.id, end="]\n")

cdef list_c* arr2list(size_t* arr, size_t n):
    cdef size_t i
    cdef list_c* l = <list_c*> malloc(sizeof(list_c) * n)

    for i in range(n - 1):
        l[i].id = arr[i]
        l[i].next = l + i + 1

    l[n - 1].id = arr[n - 1]
    l[n - 1].next = NULL
    # print_mem(<size_t *>l, 2 * n)
    return l

""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

def test_print_l_list():
    
    cdef list_c first, second, third
    first.id = 1
    second.id = 2
    third.id = 3
    first.next = &second
    second.next = &third
    third.next = NULL
    print_l(&first)

def test_create_l_list():
    
    cdef:
        size_t[3] arr = [1, 2, 3]
        list_c* l = arr2list(&arr[0], 3)
    for i in range(3):
        assert l.id == arr[i]
        l = l.next
    PyMem_Free(l)


def test_insert_l_list():
    
    cdef:
        size_t[3] arr = [1, 2, 3]
        list_c* l = arr2list(&arr[0], 3)
    insert_l(l, 4)
    for i in range(4):
        assert l.id == i + 1
        l = l.next
    PyMem_Free(l)

def test_create_l_list_random():
    DEF size = 1000
    
    cdef size_t[size] arr
    cdef list_c * l
    cdef size_t i, j

    for j in range(100):
        a = np.random.randint(0, 100, size)
        for i in range(size):
            arr[i] = a[i]
        l = arr2list(&arr[0], size)
        for i in range(size):
            assert l.id == a[i]
            l = l.next
        PyMem_Free(l)
