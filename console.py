import numpy as np

''' Python implementation of QuickSort using Hoare's
partition scheme. '''

''' This function takes first element as pivot, and places
      all the elements smaller than the pivot on the left side
      and all the elements greater than the pivot on
      the right side. It returns the index of the last element
      on the smaller side '''


def partition(arr, low, high):
    pivot = arr[low]
    i = low - 1
    j = high + 1

    while (True):

        # Find leftmost element greater than
        # or equal to pivot
        i += 1
        while arr[i] < pivot:
            i += 1

        # Find rightmost element smaller than
        # or equal to pivot
        j -= 1
        while arr[j] > pivot:
            j -= 1

        # If two pointers met.
        if i >= j:
            return j

        arr[i], arr[j] = arr[j], arr[i]


''' The main function that implements QuickSort
arr --> Array to be sorted,
low --> Starting index,
high --> Ending index '''


def quickSort(arr, low, high):
    ''' pi is partitioning index, arr[p] is now
    at right place '''
    if (low < high):
        print("input arr: ", arr)
        print(f"old_p: {low}   ", "   " * low, "↑")
        pi = partition(arr, low, high)
        print("output arr:", arr)
        print(f"new_p: {pi}   ", "   " * pi, "↑")
        print("==================================")

        # Separately sort elements before
        # partition and after partition
        quickSort(arr, low, pi)
        quickSort(arr, pi + 1, high)


''' Function to print an array '''


# This code is contributed by shubhamsingh10

n = 5
np.random.seed(3)
for i in range(10):
    arr = np.random.randint(0, 2*n, n).astype(np.float64)
    # arr = np.array([5., 7., 6., 0., 4.])
    # p_idx = n // 2
    # p_idx = 0
    # pivot = arr[p_idx]
    # print("input arr: ", arr)
    # print("old_p:     ", "   " * p_idx, "↑")
    # new_pi = partition(arr, 0, n - 1)
    # print("output arr:", arr)
    # print("new_p:     ", "   " * new_pi, "↑")
    # print("==================================")
    quickSort(arr, 0, n - 1)
    # print("sorted arr:", arr)
    # assert arr[new_pidx] == pivot
    print("==================================")
    print("============ FINISHED ============")
    print("==================================")

