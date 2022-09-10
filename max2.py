import numpy as np


def max_r(arr):
    n = len(arr)
    # base case
    if n == 2:  # T(2) = 1 comparison
        if arr[0] > arr[1]:
            return arr[0], arr[1]
        else:
            return arr[1], arr[0]

    # divide
    left = arr[:n // 2]
    right = arr[n // 2:]
    q1, q2 = max_r(left)  # T(n/2)
    r1, r2 = max_r(right)  # T(n/2)

    # conquer
    res = [q1, q2, r1, r2]
    res.sort()  # const * n
    return res[3], res[2]


def max_2r_(arr, start=0, end=0, level=0):
    """
    :param arr:
    :param start:start index
    :param end: ending index
    :return: index of maximum element and maximum from another branch
    """
    n = len(arr[start:end])
    # base case
    if n == 2:  # T(2) = 1 comparison
        if arr[start] > arr[end-1]:
            return start, end-1
        else:
            return end-1, start

    # divide
    i1, i2 = max_2r_(arr, start, start + n // 2, level+1)      # T(n/2)
    j1, j2 = max_2r_(arr, start + n // 2, start + n, level+1)  # T(n/2)

    print("\n")
    print(arr[i1], arr[j1])
    # conquer
    if arr[i1] > arr[j1]:  # C = 1 comprarison
        return i1, j1
    else:
        return j1, i1

def max_2(arr):
    tree = {}
    n = len(arr)
    max_2r_(arr, tree, 0, n)



def test_max_1():
    assert max_2r_([1, 2, 3, 4, 5, 6, 7, 8], 0, 8)[0] == 7


def test_max_2_1():
    assert max_r([1, 2, 3, 4]) == (4, 3)


def test_max_2_2():
    assert max_r(np.array([1, 2, 3, 4])) == (4, 3)


def test_max_2_3():
    for i in range(100):
        arr = np.random.randn(64)
        res = arr.copy()
        res.sort()
        assert max_r(arr.copy()) == (res[-1], res[-2])
