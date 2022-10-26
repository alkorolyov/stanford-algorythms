

def read_graph_l(filename):
    with open(filename, "r") as f:
        read_buf = f.readlines()
    # print(read_buf)
    return read_buf


#%%
path = "tests//course2_assignment2Dijkstra//"
filename = "input_random_1_4.txt"

for line in read_graph_l(path + filename)[:10]:
    line = line.replace("\n", "").replace("\r", "")
    line = line.split("\t")
    print(line)
    v = int(line[0])
    edges = line[1:]
    print("v:", v)
    print("edges")
    for e in edges:
        e = [int(s) for s in e.split(",")]
        # addedge(v, e[0], e[1])
        print(e)

#%%
import numpy as np

A = np.array([[0.8, 0.3], [0.2, 0.7]])

u0 = [0, 1]
v0 = [1, 0]

u = [np.array(u0)]
v = [np.array(v0)]

print(A)
print()
for i in range(7):
    u.append(np.matmul(A, u[i]))
    print(u[i + 1])
    print(f"{np.sum(u[i + 1]):.4f}")
print("=========")
for i in range(7):
    v.append(np.matmul(A, v[i]))
    print(v[i + 1])
    print(f"{np.sum(v[i + 1]):.4f}")

#%%
from scipy.linalg import lu
n = 1000000
res = np.empty((n, 3))
A = np.random.rand(3, 3, n)
for i in range(n):
    # A = np.random.rand(3, 3)
    P, L, U = lu(A[:, :, i])
    res[i, :] = [U[0, 0], U[1, 1], U[2, 2]]
    # print(res[i, 0])
    # print(A)
    # print(U)
# print(res)
print(res.mean(axis=0))


