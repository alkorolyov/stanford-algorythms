import numpy as np
from tqdm import trange
from quicksort import quicksort_c, read_numpy
import pstats, cProfile


def count_comparisons(arr, partition_func) -> int:
    if len(arr) <= 1:
        return 0

    # if len(arr) == 2 and arr[0] > arr[1]:
    #     swap(arr, 0, 1)
    #     return

    p_idx = partition_func(arr)
    left = arr[:p_idx]
    right = arr[p_idx + 1:]
    cmps_root = len(arr) - 1
    cmps_left = count_comparisons(left, partition_func)
    cmps_right = count_comparisons(right, partition_func)
    return cmps_root + cmps_left + cmps_right


def quicksort_py(arr, partition_func):
    if len(arr) <= 1:
        return 0
    p_idx = partition_func(arr)
    left = arr[:p_idx]
    right = arr[p_idx + 1:]
    quicksort_py(left, partition_func)
    quicksort_py(right, partition_func)
    return

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def partition_p_first(arr) -> int:
    """
    Partitions array taking pivot as first element.
    :param arr: input array
    :return: index of border between subarrays
    """
    p = arr[0]
    j = 1
    for i in range(1, len(arr)):
        if arr[i] < p:
            swap(arr, i, j)
            j += 1
    swap(arr, 0, j - 1)
    return j - 1

def partition_p_last(arr) -> int:
    swap(arr, 0, -1)
    return partition_p_first(arr)


def partition_p_median(arr) -> int:
    p_idx = get_median_three_idx(arr)
    swap(arr, 0, p_idx)
    return partition_p_first(arr)

def partition_p_random(arr) -> int:
    p_idx = np.random.randint(0, len(arr))
    swap(arr, 0, p_idx)
    return partition_p_first(arr)

def get_median_three_idx(arr):
    middle_idx = (len(arr) - 1) // 2
    return get_middle_idx([arr[0], arr[-1], arr[middle_idx]])


def get_middle_idx(els: list):
    if els[0] < els[1] < els[2]:
        return 1
    elif els[1] < els[2] < els[0]:
        return 2
    else:
      return 0


for i in trange(1000):
    arr = np.random.randn(100)
    arr_c = arr.copy()
    quicksort_c(arr)
    if not all(arr == np.sort(arr_c)):
        print("============== Error ===============")

# arr = np.random.randint(0, 10, 5).astype(np.float64)
# print(arr)
# # print(count_comparisons(arr, partition_p_random))
# print(quicksort_c(arr))
# print(arr)
# print(all(arr == np.sort(arr)))


# arr = np.random.randn(1000000)
# cProfile.runctx("quicksort_c(arr)", globals(), locals(), "Profile.prof")
# s = pstats.Stats("Profile.prof")
# s.strip_dirs().sort_stats("time").print_stats()

read_numpy(np.random.randn(100))

arr = np.random.randn(100000)
# %timeit quicksort_py(arr, partition_p_first)
# %timeit quicksort_py(arr, partition_p_last)
# %timeit quicksort_py(arr.copy(), partition_p_random)
%timeit quicksort_c(arr.copy())
%timeit np.sort(arr.copy())


#%%


#%%
check = []
for i in range(10):
    arr = np.random.randint(0, 100, 20)
    count_comparisons(arr, partition_p_random)
    check.append(all(arr == np.sort(arr)))
print(all(check))

#%%
with open("quicksort.txt", "r") as f:
    list_int = [int(s) for s in f.read().split("\n")[:-1]]
    arr = np.array(list_int, dtype=np.uint32)

print("first", count_comparisons(arr.copy(), partition_p_first))
print("last", count_comparisons(arr.copy(), partition_p_last))
print("median", count_comparisons(arr.copy(), partition_p_median))
print("random", count_comparisons(arr.copy(), partition_p_random))
#%%

arr = np.random.randint(0, 10, 5)
quicksort_py(arr, partition_p_first)
print(all(arr == np.sort(arr)))

check = []
for i in range(1000):
    arr = np.random.randint(0, 100, 20)
    quicksort_py(arr, partition_p_first)
    check.append(all(arr == np.sort(arr)))
print(all(check))
#%%
arr = np.random.randn(1000)
# %timeit quicksort_py(arr, partition_p_first)
# %timeit quicksort_py(arr, partition_p_last)
# %timeit quicksort_py(arr.copy(), partition_p_random)
%timeit quicksort_c(arr.copy())
%timeit np.sort(arr.copy())


