from setuptools import setup, Extension
from time import time
from Cython.Build import cythonize
from Cython.Compiler.Version import version as cython_version
from Cython.Compiler import Options
import numpy as np

# Cython Compiler options
import utils

Options.cimport_from_pyx = False

fast_directives = {
    "language_level": "3",
    "profile": False,
    "linetrace": False,
    "emit_code_comments": False,

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
    "annotate": False,
    "nthreads": 22,
    "compiler_directives": fast_directives
}


def mk_ext(name, files):
    return Extension(name, files,
                     include_dirs=[np.get_include()],
                     extra_compile_args=["/O2", "/Ob3", "/arch:AVX2", "/openmp"],
                     define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
                     )


if __name__ == '__main__':

    extensions = [
        mk_ext('c_utils', ['c_utils.pyx']),
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
        mk_ext('quicksort', ['quicksort.pyx']),
        mk_ext('heapsort', ['heapsort.pyx']),
        mk_ext('introsort', ['introsort.pyx']),
        mk_ext('insertsort', ['insertsort.pyx']),
        mk_ext('selection', ['selection.pyx']),
        mk_ext('mincut', ['mincut.pyx']),
        mk_ext('scc', ['scc.pyx']),
        mk_ext('dijkstra', ['dijkstra.pyx']),
        mk_ext('ht_chain', ['ht_chain.pyx']),
        mk_ext('ht_openaddr', ['ht_openaddr.pyx'])

    ]

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

    from tests.runner import run_tests
    run_tests(extensions)

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


    from utils import parse_time, timeit_func, time_sorts


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
              "n = 10000\n" \
              "arr = np.random.randn(n)\n"

    imports = "from quicksort import qsort_cy, qsort_stack\n" \
              "import numpy as np\n" \
              "n = 10000\n" \
              "arr = np.random.randn(n)\n" \
              "ar = np.random.randn(1024)\n"

            # "from introsort import introsort_py\n" \
            # "from insertsort import insertsort_py\n" \
            # "from heapsort import hsort_py\n" \

            # "arr.qsort()\n" \
            # "arr = np.flip(arr)\n" \

            # "arr.qsort()\n" \
            # "np.random.seed(0)\n" \

    # timeit_func("r_select", "arr.copy(), n // 2", imports)
    # timeit_func("d_select", "arr.copy(), n // 2", imports)
    # timeit_func("np.median", "arr.copy()", imports)

    # timeit_func("quicksort_c", "arr.copy()", imports)
    # timeit_func("np.qsort", "arr.copy(), kind='mergesort'", imports)

    # timeit_func("quicksort_c", "arr.copy()", imports)
    # timeit_func("introsort_py", "arr.copy()", imports)
    # timeit_func("insertsort_py", "ar.copy()", imports)

    # timeit_func("qsort_cy", "arr.copy()", imports)
    # timeit_func("qsort_stack", "arr.copy()", imports)
    # timeit_func("np.sort", "arr.copy(), kind='quicksort'", imports)

    # timeit_func("hsort_py", "arr.copy()", imports)
    # timeit_func("np.qsort", "arr.copy(), kind='heapsort'", imports)


    # timeit_func("min_dist_c", "arr.copy()", imports)
    # timeit_func("min_dist_naive", "arr.copy()", imports)
    # timeit_func("pdist", "arr.copy()", imports, ".min()")
    # timeit_func("min_dist32", "arr.copy().astype(np.float32)", imports)

    # timeit_func("mincut.mincut_n", "graph, 1, mem_mode=1", imports)
    # timeit_func("mincut_py.mincut_n", "graph, 1", imports)

    # imports = "from graph import rand_graph_l_py"
    # timeit_func("rand_graph_l_py", "100, 10000", imports)
    #
    # imports = "from c_utils import rand_py, frand_py, frand32_py"
    # timeit_func("rand_py", "", imports)
    # timeit_func("frand_py", "", imports)
    # timeit_func("frand32_py", "", imports)


    # utils.NUM_RUNS = 21
    # for i in range(1, 8):
    #     time_sorts(10 ** i)


    # heap_c.time_log2()

    # dijkstra.time_naive()
    # dijkstra.time_heap()