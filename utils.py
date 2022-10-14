import inspect
from io import StringIO
import sys

saved_stdout = None

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
    print(inspect.stack()[1].code_context[0].strip('\n'), end=end)
