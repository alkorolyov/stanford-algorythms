from setuptools import setup, Extension
from time import time
from Cython.Build import cythonize
import pstats, cProfile
import numpy as np

ext_options = {}

extensions = [Extension('closestpair',
                        sources=['closestpair.pyx'],
                        include_dirs=[np.get_include()],
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
                        ),
              Extension('sorting',
                        sources=['sorting.pyx'],
                        include_dirs=[np.get_include()],
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
                        ),
              Extension('selection',
                        sources=['selection.pyx'],
                        include_dirs=[np.get_include()],
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
                        )
              ]

setup(
    name='algorithms',
    version='1.0',
    author='ergot',
    ext_modules=cythonize(extensions, **ext_options),
    test_suite='tests',
    tests_require=['pytest']
)

print("============================ UNIT TESTS ===================================")
start_time = time()
import sorting
import selection
import closestpair

sorting.test_swap_c()
sorting.test_partition_c_1()
sorting.test_partition3_c_1()
sorting.test_qsort_c_1()
sorting.test_qsort_c_2()

sorting.test_merge_c_1()
sorting.test_merge_c_2()
sorting.test_merge_c_3()
sorting.test_merge_c_4()
sorting.test_merge_c_5()
sorting.test_merge_c_6()
sorting.test_merge_c_7()
sorting.test_merge_c_8()

sorting.test_msort_c_1()
sorting.test_msort_c_2()
sorting.test_msort_c_3()

selection.test_r_select_c_1()
selection.test_r_select_c_2()
selection.test_r_select_c_3()
selection.test_r_select_c_4()


selection.test_median5_1()
selection.test_median5_2()
selection.test_median5_3()
selection.test_median5_4()

selection.test_median_c_1()
selection.test_median_c_2()
selection.test_median_c_3()
selection.test_median_c_4()

selection.test_d_select_1()
selection.test_d_select_2()
selection.test_d_select_3()
selection.test_d_select_4()
selection.test_d_select_5()
selection.test_d_select_6()

closestpair.test_min_dist_naive_c_1()

closestpair.test_min_dist_c_1()
closestpair.test_min_dist_c_2()
closestpair.test_min_dist_c_3()

closestpair.test_mindist32_1()



print(f"PASSED {time() - start_time:.2f}s")

print("============================ PROFILING ====================================")
from sorting import quicksort_c, mergesort_c

from selection import d_select, r_select
from closestpair import min_dist_c

# arr = np.random.randn(100000)
# cProfile.runctx("quicksort_c(arr)", globals(), locals(), "Profile.prof")

# arr = np.random.randn(10000000)
# cProfile.runctx("r_select(arr, arr.shape[0] // 2)", globals(), locals(), "Profile.prof")

arr = np.random.randn(100000, 2)
cProfile.runctx("min_dist_c(arr)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

print("============================ TIMING =======================================")

from timeit import Timer

def parse_time(time: float) -> str:
    if time < 1e-6:
        return f"{time / 1e-9:.2f} ns"
    elif time < 1e-3:
        return f"{time / 1e-6:.2f} µs"
    elif time < 1.0:
        return f"{time / 1e-3:.2f} ms"
    else:
        return f"{time:.2f} s"

def timeit_func(func, arg_string: str, import_string: str, post_string: str=""):
    t = Timer(stmt=f"{func}({arg_string}){post_string}",
                   setup=import_string)
    NUM_LOOPS = t.autorange()[0] // 5
    NUM_RUNS = 3
    result = np.array(t.repeat(repeat=NUM_RUNS, number=NUM_LOOPS))
    run_time = result.mean() / NUM_LOOPS
    std = result.std() / NUM_LOOPS
    print(f"{func:20s} {parse_time(run_time):8s} ± {parse_time(std):8s} (of {NUM_RUNS} runs {NUM_LOOPS:.0f} loops each)")

imports = "from sorting import quicksort_c, quicksort_mv, mergesort_c\n" \
          "from selection import r_select, d_select\n" \
          "from closestpair import min_dist_naive_mv, min_dist_naive, min_dist_mv, min_dist_c, max_points_c, min_dist32\n" \
          "import numpy as np\n" \
          "from scipy.spatial.distance import pdist\n" \
          "n = 100000\n" \
          "arr = np.random.randn(n)\n"


timeit_func("r_select", "arr.copy(), n // 2", imports)
timeit_func("d_select", "arr.copy(), n // 2", imports)
timeit_func("np.median", "arr.copy()", imports)

# timeit_func("quicksort_c", "arr.copy()", imports)
# timeit_func("np.sort", "arr.copy(), kind='mergesort'", imports)
#
# timeit_func("quicksort_c", "arr.copy()", imports)
# timeit_func("np.sort", "arr.copy(), kind='quicksort'", imports)

# timeit_func("min_dist_c", "arr.copy()", imports)
# timeit_func("min_dist_naive", "arr.copy()", imports)
# timeit_func("pdist", "arr.copy()", imports, ".min()")
# timeit_func("min_dist32", "arr.copy().astype(np.float32)", imports)




