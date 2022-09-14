import numpy as np

def max1_r(arr):
    """
    recursive maximum calculation
    :param arr: input array
    :return: maximum
    """
    n = len(arr)
    # base case
    if n == 2:  # T(2) = 1 comparison
        return max(arr[0], arr[1])

    # divide
    left = arr[:n // 2]
    right = arr[n // 2:]
    q = max1_r(left)  # T(n/2)
    r = max1_r(right)  # T(n/2)

    # conquer
    return max(q, r)    # 1 comparison

def _max2r(arr, cmp_list=[]):
    """
    Recursion call for 2nd maximum calculation
    :param arr: input array
    :param cmp_list: list of all pairwise comparisons
    :return: maximum
    """
    n = len(arr)
    # base case
    if n == 2:  # T(2) = 1 comparison
        cmp_list.append([arr[0], arr[1]])
        return max(arr[0], arr[1])

    # divide
    left = arr[:n // 2]
    right = arr[n // 2:]
    q = _max2r(left, cmp_list)  # T(n/2)
    r = _max2r(right, cmp_list)  # T(n/2)

    # conquer
    cmp_list.append([q, r])
    return max(q, r)

def max2(arr):
    """
    2nd maximum calculation
    In a tree of recursive calls the second maximum is always among first leaves
    of the path containing first maximum. On each recursion level there is
    a single comparison, totaling log2(n) - 1 additional comparisons.
    :param arr: input array
    :return: second maximum
    """
    pairs = []
    max_1 = _max2r(arr, pairs)  # total n - 1 recursive comparisons
    max2_candidates = []

    # filter pairs with first maximum only
    for p in pairs:
        if max_1 in p:
            idx = p.index(max_1)
            p.pop(idx)
            max2_candidates.append(p[0])
    return max(max2_candidates)     # log2(n) - 1 = recursion depth comparisons

def test_max_1():
    # print("\n")
    assert _max2r([1, 2, 3, 4, 5, 6, 7, 8]) == 8

def test_max_2():
    # print("\n")
    assert _max2r([4, 2, 1, 6, 8, 5, 3, 7]) == 8

def test_max_3():
    # print("\n")
    assert max2([4, 2, 1, 6, 8, 5, 3, 7]) == 7

def test_max_4():
    assert max1_r([1, 2, 3, 4]) == 4

def test_max_5():
    assert max1_r(np.array([1, 2, 3, 4])) == 4


def test_max_6():
    for i in range(100):
        arr = np.random.randn(64)
        res = arr.copy()
        res.sort()
        assert max1_r(arr.copy()) == res[-1]

def test_max_7():
    for i in range(100):
        arr = np.random.randn(64)
        res = arr.copy()
        res.sort()
        assert max2(arr.copy()) == res[-2]
