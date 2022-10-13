from setuptools import setup, Extension
from time import time
from Cython.Build import cythonize
import numpy as np

ext_options = {"annotate": True}

extensions = [Extension('array_c',
                        sources=['array_c.pyx'],
                        include_dirs=[np.get_include()],
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
                        ),
              Extension('stack',
                        sources=['stack.pyx'],
                        include_dirs=[np.get_include()],
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
                        ),
              Extension('queue_c',
                        sources=['queue_c.pyx'],
                        include_dirs=[np.get_include()],
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
                        ),
              Extension('graph',
                        sources=['graph.pyx'],
                        include_dirs=[np.get_include()],
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
                        ),
              Extension('dfs',
                        sources=['dfs.pyx'],
                        include_dirs=[np.get_include()],
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
                        ),
              Extension('bfs',
                        sources=['bfs.pyx'],
                        include_dirs=[np.get_include()],
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
                        ),

              Extension('closestpair',
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
                        ),
              Extension('mincut',
                        sources=['mincut.pyx'],
                        include_dirs=[np.get_include()],
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
                        ),
              Extension('scc',
                        sources=['scc.pyx'],
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
import mincut
import stack
import array_c
import graph
import dfs
import bfs
import queue_c

stack.test_push()
stack.test_print()
stack.test_empty()
stack.test_full()
stack.test_size()
stack.test_random()

queue_c.test_enqueue()
queue_c.test_empty()
queue_c.test_full()
queue_c.test_print()

array_c.test_list2arr()
array_c.test_create_arr()
array_c.test_resize_arr()
array_c.test_print()
array_c.test_print_zero_length()

graph.test_create_graph()
graph.test_add_edge()
graph.test_dict2graph()
graph.test_dict2graph_1()
graph.test_dict2graph_2()
graph.test_dict2graph_random()

bfs.test_bfs()

dfs.test_dfs_1()
dfs.test_dfs_2()
dfs.test_dfs_3()
dfs.test_dfs_4()
dfs.test_dfs_random()


# sorting.test_swap_c()
# sorting.test_partition_c_1()
# sorting.test_partition3_c_1()
# sorting.test_qsort_c_1()
# sorting.test_qsort_c_2()
#
# sorting.test_merge_c_1()
# sorting.test_merge_c_2()
# sorting.test_merge_c_3()
# sorting.test_merge_c_4()
# sorting.test_merge_c_5()
# sorting.test_merge_c_6()
# sorting.test_merge_c_7()
# sorting.test_merge_c_8()
#
# sorting.test_msort_c_1()
# sorting.test_msort_c_2()
# sorting.test_msort_c_3()
# print()
#
# selection.test_r_select_c_1()
# selection.test_r_select_c_2()
# selection.test_r_select_c_3()
# selection.test_r_select_c_4()
#
#
# selection.test_median5_1()
# selection.test_median5_2()
# selection.test_median5_3()
# selection.test_median5_4()
#
# selection.test_median_c_1()
# selection.test_median_c_2()
# selection.test_median_c_3()
# selection.test_median_c_4()
#
# selection.test_d_select_1()
# selection.test_d_select_2()
# selection.test_d_select_3()
# selection.test_d_select_4()
# selection.test_d_select_5()
# selection.test_d_select_6()
#
# closestpair.test_min_dist_naive_c_1()
#
# closestpair.test_min_dist_c_1()
# closestpair.test_min_dist_c_2()
# closestpair.test_min_dist_c_3()
#
# closestpair.test_mindist32_1()
#
# mincut.test_create_graph()
# mincut.test_read_graph_c_1()
# mincut.test_read_graph_c_2()
# mincut.test_read_graph_c_3()
# mincut.test_read_graph_c_4()
# mincut.test_read_graph_c_random()
#
# mincut.test_copy_graph()
# mincut.test_random_pair()
# mincut.test_pop_from_graph()
# mincut.test_pop_from_graph_1()
# mincut.test_delete_self_loops()
# mincut.test_transfer_vertices()
# mincut.test_delete_vertex()
# mincut.test_delete_vertex_1()
# mincut.test_replace_references()
# mincut.test_contract()
# mincut.test_mincut()
# mincut.test_mincut_1()
# mincut.test_mincut_N()

# scc.test_ascii2int()
# scc.test_read_edge_1()
# scc.test_read_buf_1()
# scc.test_create_l_list()
# scc.test_create_l_list_random()
# scc.test_insert_l_list()
# scc.test_print_l_list()
# scc.test_read_arr()
# scc.test_create_graph()
# scc.test_read_graph()
# scc.test_read_graph_1()
# scc.test_read_graph_random()
# scc.test_reverse_graph()
# scc.test_dfs_1()
# scc.test_dfs_2()
# scc.test_dfs_3()
# scc.test_dfs_4()
# scc.test_dfs_random()
# scc.test_dfs_loop_1()
# scc.test_dfs_loop_2()
# scc.test_scc_1()
# scc.test_scc_2()
# scc.test_scc_3()
# scc.test_scc_4()
# scc.test_dfs_big()
# scc.test_scc_big()


print(f"PASSED {time() - start_time:.2f}s")

print("============================ PROFILING ====================================")

# graph = gen_random_graph(200, 3000)
# cProfile.runctx("mincut_n(graph, 200, mem_mode=0)", globals(), locals(), "Profile.prof")

# arr = np.random.randn(100000)
# cProfile.runctx("quicksort_c(arr)", globals(), locals(), "Profile.prof")

# arr = np.random.randn(10000000)
# cProfile.runctx("r_select(arr, arr.shape[0] // 2)", globals(), locals(), "Profile.prof")

# arr = np.random.randn(100000, 2)
# cProfile.runctx("min_dist_c(arr)", globals(), locals(), "Profile.prof")
#
# s = pstats.Stats("Profile.prof")
# s.strip_dirs().sort_stats("time").print_stats()

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
          "import mincut\n" \
          "import mincut_py\n" \
          "import numpy as np\n" \
          "from scipy.spatial.distance import pdist\n" \
          "graph = mincut.read_file()"
          # "n = 100000\n" \
          # "arr = np.random.randn(n)\n"


# timeit_func("r_select", "arr.copy(), n // 2", imports)
# timeit_func("d_select", "arr.copy(), n // 2", imports)
# timeit_func("np.median", "arr.copy()", imports)

# timeit_func("quicksort_c", "arr.copy()", imports)
# timeit_func("np.sort", "arr.copy(), kind='mergesort'", imports)
#
# timeit_func("quicksort_c", "arr.copy()", imports)
# timeit_func("np.sort", "arr.copy(), kind='quicksort'", imports)

# timeit_func("min_dist_c", "arr.copy()", imports)
# timeit_func("min_dist_naive", "arr.copy()", imports)
# timeit_func("pdist", "arr.copy()", imports, ".min()")
# timeit_func("min_dist32", "arr.copy().astype(np.float32)", imports)

# timeit_func("mincut.mincut_n", "graph, 1, mem_mode=1", imports)
# timeit_func("mincut_py.mincut_n", "graph, 1", imports)


# start_time = time()
# graph, g_rev = scc.read_file()
# print(f"{time() - start_time:.2f}s")
# print(len(graph), len(g_rev))
# print(graph)

# start_time = time()
# scc.test_read_graph_1(graph)
# print(f"{time() - start_time:.2f}s")

# input("Press Enter to continue...")

