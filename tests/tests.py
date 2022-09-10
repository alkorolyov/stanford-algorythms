from closestpair_py import min_dist_naive_py, min_dist_py
from closestpair import min_dist_naive, min_dist_naive_mv, min_dist_mv, min_dist_c, max_points_c
from sorting import mergesort_c, quicksort_c, read_numpy
from selection import r_select, d_select
import numpy as np

def test_min_dist_py():
    for i in range(10):
        arr = np.random.randn(10, 2)
        assert np.abs(min_dist_py(arr) - min_dist_naive_py(arr)) < 1e-16

def test_min_dist_c():
    for i in range(100):
        arr = np.random.randn(10, 2)
        assert np.abs(min_dist_c(arr) - min_dist_py(arr)) < 1e-16

def test_min_dist_mv():
    for i in range(100):
        arr = np.random.randn(10, 2)
        assert np.abs(min_dist_mv(arr) - min_dist_c(arr)) < 1e-16


def test_min_dist_naive_mv():
    for i in range(100):
        arr = np.random.randn(100, 2)
        assert np.abs(min_dist_naive_mv(arr) - min_dist_c(arr)) < 1e-16

def test_min_dist_naive():
    for i in range(100):
        arr = np.random.randn(100, 2)
        assert np.abs(min_dist_naive(arr) - min_dist_c(arr)) < 1e-16


def test_max_points_c():
    for i in range(100):
        arr = np.random.randn(100, 2)
        assert np.abs(max_points_c(arr)[1] - min_dist_c(arr)) < 1e-16

def test_read_numpy_c():
        arr = np.random.randn(100)
        assert arr[0] == read_numpy(arr)

def test_quicksort_c_float():
    for i in range(100):
        arr = np.random.randn(100)
        arr_c = arr.copy()
        quicksort_c(arr)
        assert np.all(abs(arr - np.sort(arr_c)) < 1e-16)

def test_quicksort_c_dup():
    for i in range(100):
        arr = np.random.randint(0, 100, 50).astype(np.float64)
        arr_c = arr.copy()
        quicksort_c(arr)
        assert np.all(abs(arr - np.sort(arr_c)) < 1e-16)

def test_mergesort_c():
    for i in range(100):
        arr = np.random.randn(100)
        arr_c = arr.copy()
        mergesort_c(arr)
        assert np.all(np.abs(arr - np.sort(arr_c)) < 1e-16)


def test_mergesort_c_dup():
    for i in range(100):
        arr = np.random.randint(0, 100, 50).astype(np.float64)
        arr_c = arr.copy()
        mergesort_c(arr)
        assert np.all(np.abs(arr - np.sort(arr_c)) < 1e-16)

def test_r_select_c():
    for i in range(100):
        arr = np.random.randn(100)
        arr_c = arr.copy()
        k = np.random.randint(100) + 1
        assert r_select(arr, k) == np.sort(arr_c)[k - 1]

def test_d_select_c():
    for i in range(100):
        arr = np.random.randn(100)
        arr_c = arr.copy()
        k = np.random.randint(100) + 1
        assert d_select(arr, k) == np.sort(arr_c)[k - 1]
