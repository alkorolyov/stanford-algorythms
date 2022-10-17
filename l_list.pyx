# cython: language_level=3

from libc.stdlib cimport malloc, free, rand
from utils import print_func_name

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

""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

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
