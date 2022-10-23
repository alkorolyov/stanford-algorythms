

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

