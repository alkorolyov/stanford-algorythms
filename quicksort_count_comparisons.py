import numpy as np
from tqdm import trange

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

#%%
for i in range(10):
    arr = np.random.randint(0, 100, 20)
    count_comparisons(arr, partition_p_random)
    assert all(arr == np.sort(arr))

#%%
with open("quicksort.txt", "r") as f:
    list_int = [int(s) for s in f.read().split("\n")[:-1]]
    arr = np.array(list_int, dtype=np.uint32)

print("first", count_comparisons(arr.copy(), partition_p_first))
print("last", count_comparisons(arr.copy(), partition_p_last))
print("median", count_comparisons(arr.copy(), partition_p_median))
print("random", count_comparisons(arr.copy(), partition_p_random))
