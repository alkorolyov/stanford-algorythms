import inspect
from io import StringIO
import sys
from timeit import Timer
import numpy as np

saved_stdout = None

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
    NUM_LOOPS = t.autorange()[0]
    NUM_RUNS = 7
    result = np.array(t.repeat(repeat=NUM_RUNS, number=NUM_LOOPS))
    run_time = result.mean() / NUM_LOOPS
    std = result.std() / NUM_LOOPS
    print(f"{func:20s} {parse_time(run_time):8s} ± {parse_time(std):8s} (of {NUM_RUNS} runs {NUM_LOOPS:.0f} loops each)")


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
        print(inspect.stack()[1].code_context[0].strip('\n'), end=end)
    except TypeError:
        print("'None' in function call stack")


def iterable(py_obj: object) -> bool:
    try:
        iter(py_obj)
        return True
    except TypeError:
        return False
