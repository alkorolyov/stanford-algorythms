import inspect
from io import StringIO
import sys
from timeit import Timer
import numpy as np

saved_stdout = None

def f8(x):
    # pstats.f8 = f8
    ret = "%8.3f" % x
    if ret != '   0.000':
        return ret
    ret = "%6dµs" % (x * 1000000)
    if x > 1e-5:
    # if ret != '     0µs':
        return ret
    ret = "%6.2fµs" % (x * 1000000)
    return ret

def parse_time(time: float) -> str:
    if time < 1e-6:
        return f"{time / 1e-9:.2f} ns"
    elif time < 1e-3:
        return f"{time / 1e-6:.2f} µs"
    elif time < 1.0:
        return f"{time / 1e-3:.2f} ms"
    else:
        return f"{time:.2f} s"


NUM_RUNS = 21
NUM_LOOPS_DIV = 1

def timeit_func(func, arg_string: str, import_string: str, post_string: str=""):
    t = Timer(stmt=f"{func}({arg_string}){post_string}",
                   setup=import_string)
    global NUM_RUNS
    global NUM_LOOPS_DIV
    NUM_LOOPS = t.autorange()[0] // NUM_LOOPS_DIV
    result = np.array(t.repeat(repeat=NUM_RUNS, number=NUM_LOOPS))
    run_time = result.mean() / NUM_LOOPS
    std = result.std() / NUM_LOOPS
    print(f"{func:20s} {parse_time(run_time):8s} ± {parse_time(std):8s} (of {NUM_RUNS} runs {NUM_LOOPS:.0f} loops each)")

def time_sorts(arr_len):
    imports = "from quicksort import qsort_cy, qsort_cmp_py\n" \
              "import numpy as np\n" \
              f"n = {arr_len}\n" \
              "arr = np.random.randn(n)\n"

    print(f"Array length: {arr_len}")
    timeit_func("qsort_cy", "arr.copy()", imports)
    timeit_func("qsort_cmp_py", "arr.copy()", imports)
    timeit_func("np.sort", "arr.copy(), kind='quicksort'", imports)


def set_stdout():
    s = StringIO()
    global saved_stdout
    saved_stdout = sys.stdout
    sys.stdout = s
    return s


def restore_stdout():
    global saved_stdout
    sys.stdout = saved_stdout


def print_func_name(end="\n"):
    try:
        print(inspect.stack()[1].code_context[0].strip('\n').strip(' '), end=end)
    except TypeError:
        print("'None' in function call stack")


def iterable(py_obj: object) -> bool:
    try:
        iter(py_obj)
        return True
    except TypeError:
        return False
