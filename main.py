import numpy as np

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

def time_func(t: Timer) -> float:
    result = t.autorange()
    return result[1] / result[0]


def parse_function(statement: str, setup: str = ""):
    func_name = statement.split("(")[0]
    if func_name.endswith("_py"):
        t = Timer(stmt=statement,
                  setup=f"from closestpair_py import {func_name}\n{setup}")
        run_time = time_func(t)
        print(f"{func_name:30s} {parse_time(run_time)}")
    else:
        t = Timer(stmt=statement,
                  setup=f"from closestpair import {func_name}\n{setup}")
        run_time = time_func(t)
        print(f"{func_name:30s} {parse_time(run_time)}")


if __name__ == "__main__":
    # parse_function("distance(0.1, 0.2, 0.3, 0.4)")
    # parse_function("distance_point(p1, p2)", "import numpy as np\np1, p2=np.array([.1,.2]),np.array([.3,.4])")
    # parse_function("distance_py(0.1, 0.2, 0.3, 0.4)")

    # parse_function("test_min_out_of_3_opt(P)", "import numpy\nP=numpy.random.randn(100000,3)")
    # parse_function("test_min_out_of_3(P)", "import numpy\nP=numpy.random.randn(100000,3)")

    # parse_function("min_dist_naive_py(P)", "import numpy\nP=numpy.random.randn(1000,2)")
    # parse_function("min_dist_naive(P)", "import numpy\nP=numpy.random.randn(1000,2)")
    print("")
    parse_function("min_dist_py(P)", "import numpy\nP=numpy.random.randn(10000,2)")
    parse_function("min_dist(P)", "import numpy\nP=numpy.random.randn(10000,2)")
    parse_function("min_dist_c(P)", "import numpy\nP=numpy.random.randn(10000,2)")




