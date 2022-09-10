import multiprocessing
import numpy as np
from closestpair import max_points_c
from time import time

def task(id, mp, non_eq):
    n = 3600
    for i in range(n):
        P = np.random.randn(100000, 2)
        # mp.append(max_points_c(P, strict=True)[0])
        mp1, d1 = max_points_c(P, strict=True)
        mp2, d2 = max_points_c(P, strict=False)
        mp.append(mp1)
        non_eq.append(d1 != d2)
        if id == 0 and i % 10 == 0:
            print(f"{i} / {n} Max points: {max(mp)} Checks: {any(non_eq)}")

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    mp = manager.list()
    non_eq = manager.list()
    jobs = []
    start_time = time()

    for i in range(24):
        p = multiprocessing.Process(target=task, args=(i, mp, non_eq))
        p.start()
        jobs.append(p)

    for p in jobs:
        p.join()

    print(f"Max points: {max(mp)}")
    print(f"Total time: {time() - start_time:.2f}s")
