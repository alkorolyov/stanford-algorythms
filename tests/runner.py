from time import time
from utils import set_stdout, restore_stdout


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _discover_tests(ext_name: str) -> list:
    exec(f"import {ext_name}")
    ext = locals()[ext_name]
    return [s for s in dir(ext) if s.startswith("test_")]


def test_extension(ext_name: str):
    tests = _discover_tests(ext_name)
    if tests:
        exec(f"import {ext_name}")
        for test_name in tests:
            full_name = ext_name + '.' + test_name
            start = time()
            print(f"{full_name:50s}", end="")
            try:
                exec(f"{full_name}()")
            except AssertionError:
                fail_red = bcolors.FAIL + "FAIL" + bcolors.ENDC
                print(f"[ {fail_red}  {time() - start:.2f}s ]")
            else:
                ok_green = bcolors.OKGREEN + "OK" + bcolors.ENDC
                print(f"[ {ok_green}    {time() - start:.2f}s ]")


def run_tests(extensions):
    start_time = time()
    for ext in extensions:
        test_extension(ext.name)
    print(f"PASSED {time() - start_time:.2f}s")




