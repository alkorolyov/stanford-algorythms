from setuptools import setup, Extension
from time import time
from Cython.Build import cythonize
from Cython.Compiler.Version import version as cython_version
from Cython.Compiler import Options
import numpy as np

# Cython Compiler options
Options.cimport_from_pyx = True

fast_directives = {
    "language_level": "3",
    "profile": False,
    "linetrace": False,

    "overflowcheck.fold": True,
    "boundscheck": False,
    "wraparound": False,
    "initializedcheck": False,
    "cdivision": True

    # "boundscheck": True,
    # "wraparound": True,
    # "initializedcheck": True,
    # "cdivision": False
}

ext_options = {
    "force": False,
    "annotate": True,
    "nthreads": 24,
    "compiler_directives": fast_directives
}


def mk_ext(name, files):
    return Extension(name, files,
                     include_dirs=[np.get_include()],
                     extra_compile_args=["/O2", "/Ob3", "/arch:AVX2"],
                     define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
                     )


if __name__ == '__main__':

    extensions = [mk_ext('c_utils', ['c_utils.pyx']),
                  mk_ext('array_c', ['array_c.pyx']),
                  mk_ext('stack', ['stack.pyx']),
                  mk_ext('queue_c', ['queue_c.pyx']),
                  mk_ext('heap_c', ['heap_c.pyx']),
                  mk_ext('heap_ex', ['heap_ex.pyx']),
                  mk_ext('graph', ['graph.pyx']),
                  mk_ext('readg', ['readg.pyx']),
                  mk_ext('dfs', ['dfs.pyx']),
                  mk_ext('bfs', ['bfs.pyx']),
                  mk_ext('topsort', ['topsort.pyx']),
                  mk_ext('closestpair', ['closestpair.pyx']),
                  mk_ext('sorting', ['sorting.pyx']),
                  mk_ext('selection', ['selection.pyx']),
                  mk_ext('mincut', ['mincut.pyx']),
                  mk_ext('scc', ['scc.pyx']),
                  mk_ext('dijkstra', ['dijkstra.pyx'])]

    print(f"Cython version {cython_version}")

    setup(
        name='algorithms',
        version='1.0',
        author='ergot',
        ext_modules=cythonize(extensions, **ext_options),
        test_suite='tests',
        tests_require=['pytest']
    )

    print("============================ UNIT TESTS ===================================")

    import sorting
    import selection
    import closestpair
    import mincut
    import c_utils
    import stack
    import array_c
    import queue_c
    import heap_c
    import heap_ex
    import graph
    import dfs
    import bfs
    import readg
    import topsort
    import scc
    import dijkstra

    start_time = time()

    c_utils.test_frand()
    c_utils.test_srand()
    c_utils.test_frand32()

    # stack.test_push()
    # stack.test_print()
    # stack.test_empty()
    # stack.test_full()
    # stack.test_size()
    # stack.test_random()
    #
    # queue_c.test_enqueue()
    # queue_c.test_empty()
    # queue_c.test_full()
    # queue_c.test_print()
    #
    # array_c.test_create_arr()
    # array_c.test_resize_arr()
    # array_c.test_list2arr()
    # array_c.test_arr2numpy()
    # array_c.test_swap()
    # array_c.test_reverse_even()
    # array_c.test_reverse_odd()
    # array_c.test_print()
    # array_c.test_print_zero_length()
    #
    # heap_c.test_log2()
    # heap_c.test_get_parent()
    # heap_c.test_get_children()
    # heap_c.test_create()
    # heap_c.test_heapify()
    # # heap_c.test_heapify_rnd()
    # heap_c.test_resize()
    # heap_c.test_pop_heap()
    # heap_c.test_heap_rnd()
    # heap_c.test_print_tree()
    #
    # heap_ex.test_create_ex()
    # heap_ex.test_swap_ex()
    # heap_ex.test_print_ex()
    # heap_ex.test_push_heap()
    # heap_ex.test_resize()
    # heap_ex.test_isin()
    # heap_ex.test_find()
    # heap_ex.test_pop_heap()
    # heap_ex.test_pop_heap_single()
    # heap_ex.test_replace()
    # heap_ex.test_push_pop_rnd()
    #
    # graph.test_create_graph()
    # graph.test_add_edge()
    # graph.test_dict2graph()
    # graph.test_dict2graph_1()
    # graph.test_dict2graph_2()
    # graph.test_dict2graph_random()
    # graph.test_reverse_graph()
    #
    # readg.test_ascii2int()
    # readg.test_ascii2int_1()
    # readg.test_ascii2int_2()
    # readg.test_read_edge_spc_n()
    # readg.test_read_edge_spc_rn()
    # readg.test_read_edge_rn()
    # readg.test_read_edge_n()
    # readg.test_read_buf_1()
    # readg.test_read_array()
    # readg.test_read_graph()
    # # readg.test_read_big()
    # # readg.test_read_big_pair()
    # readg.test_read_graph_l()
    #
    # # bfs.test_bfs()
    #
    # dfs.test_dfs_1()
    # dfs.test_dfs_2()
    # dfs.test_dfs_3()
    # dfs.test_dfs_4()
    # dfs.test_dfs_random()
    # # dfs.test_dfs_big()
    # # dfs.test_dfs_loop_big()
    #
    # topsort.test_topsort()
    # topsort.test_graphlib()
    # topsort.test_topsort_rnd()
    # # topsort.test_big()
    #
    # scc.test_scc_1()
    # scc.test_scc_2()
    # scc.test_scc_3()
    # scc.test_scc_4()
    # scc.test_single_case()
    # # scc.test_all_casses()
    # # scc.test_scc_big()
    #
    # dijkstra.test_naive()
    # dijkstra.test_naive_loops()
    # dijkstra.test_naive_self_loops()
    # dijkstra.test_naive_non_conn()
    # dijkstra.test_naive_zero_conn()
    # dijkstra.test_naive_empty()
    # dijkstra.test_naive_rnd()
    # dijkstra.test_naive_1()
    # dijkstra.test_single_case_naive()
    # # dijkstra.test_all_cases_naive()
    #
    # dijkstra.test_heap()
    # dijkstra.test_heap_loops()
    # dijkstra.test_heap_self_loops()
    # dijkstra.test_heap_non_conn()
    # dijkstra.test_heap_zero_conn()
    # dijkstra.test_heap_empty()
    # dijkstra.test_heap_rnd()
    #
    # dijkstra.test_single_case_heap()
    # # dijkstra.test_all_cases_heap()
    #
    # sorting.test_swap_c()
    # sorting.test_choose_p_rnd()
    # sorting.test_partition_c_1()
    # sorting.test_partition3_c_1()
    # sorting.test_partition_c_dups()
    # sorting.test_qsort_c_1()
    # sorting.test_qsort_c_dups()
    # sorting.test_qsort_c_rnd()
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
    # selection.test_r_select_c_dups()
    # selection.test_r_select_c_rnd()
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

    print(f"PASSED {time() - start_time:.2f}s")

    print("============================ PROFILING ====================================")

    import cProfile
    import pstats
    from utils import f8
    pstats.f8 = f8

    # graph = gen_random_graph(200, 3000)
    # cProfile.runctx("mincut_n(graph, 200, mem_mode=0)", globals(), locals(), "Profile.prof")

    # arr = np.random.randn(100000)
    # cProfile.runctx("quicksort_c(arr)", globals(), locals(), "Profile.prof")

    # arr = np.random.randn(10000000)
    # cProfile.runctx("r_select(arr, arr.shape[0] // 2)", globals(), locals(), "Profile.prof")

    # arr = np.random.randn(100000, 2)
    # cProfile.runctx("min_dist_c(arr)", globals(), locals(), "Profile.prof")


    # cProfile.runctx("dijkstra.test_heap_rnd()", globals(), locals(), "Profile.pstat")
    # s = pstats.Stats("Profile.pstat")
    # s.strip_dirs().sort_stats("time").print_stats()

    print("============================ TIMING =======================================")


    from utils import parse_time, timeit_func


    # imports = "from sorting import quicksort_c, quicksort_mv, mergesort_c\n" \
    #           "from selection import r_select, d_select\n" \
    #           "from closestpair import min_dist_naive_mv, min_dist_naive, min_dist_mv, min_dist_c, max_points_c, min_dist32\n" \
    #           "import mincut\n" \
    #           "import mincut_py\n" \
    #           "import numpy as np\n" \
    #           "from scipy.spatial.distance import pdist\n" \
    #           "graph = mincut.read_file()\n" \
    #           "n = 100000\n" \
    #           "arr = np.random.randn(n)\n"

    imports = "from sorting import quicksort_c, quicksort_mv, mergesort_c\n" \
              "from graph import rand_graph_l_py\n" \
              "import numpy as np\n" \
              "n = 100000\n" \
              "arr = np.random.randn(n)\n"

    # timeit_func("r_select", "arr.copy(), n // 2", imports)
    # timeit_func("d_select", "arr.copy(), n // 2", imports)
    # timeit_func("np.median", "arr.copy()", imports)

    # timeit_func("quicksort_c", "arr.copy()", imports)
    # timeit_func("np.sort", "arr.copy(), kind='mergesort'", imports)

    # timeit_func("quicksort_c", "arr.copy()", imports)
    # timeit_func("np.sort", "arr.copy(), kind='quicksort'", imports)

    # timeit_func("min_dist_c", "arr.copy()", imports)
    # timeit_func("min_dist_naive", "arr.copy()", imports)
    # timeit_func("pdist", "arr.copy()", imports, ".min()")
    # timeit_func("min_dist32", "arr.copy().astype(np.float32)", imports)

    # timeit_func("mincut.mincut_n", "graph, 1, mem_mode=1", imports)
    # timeit_func("mincut_py.mincut_n", "graph, 1", imports)

    timeit_func("rand_graph_l_py", "100, 10000", imports)

    c_utils.time_rand()

    # heap_c.time_log2()

    # dijkstra.time_naive()
    # dijkstra.time_heap()
