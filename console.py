def merge(a, b):
    i = 0
    j = 0
    c = []
    while len(c) < len(a) + len(b):
        if i == len(a):
            c.append(b[j])
            j += 1
            continue

        if j == len(b):
            c.append(a[i])
            i += 1
            continue

        if a[i] < b[j]:
            c.append(a[i])
            i += 1
        elif a[i] > b[j]:
            c.append(b[j])
            j += 1
    return c

def merge_sort(a):
    n = len(a)
    if n < 2:
        return a
    #partition
    q = a[:n // 2]
    r = a[n // 2:]
    q_sort = merge_sort(q)
    r_sort = merge_sort(r)
    res = merge(q_sort, r_sort)
    print(res)
    return res
    # return merge(q_sort, r_sort)

# print(merge([1], [2, 4, 6]))
print(merge_sort([6, 5, 4, 3, 2, 1]))



def partition(a, p):
    q, r = [], []
    for i in range(len(a)):
        if a[i] < p:
            q.append(a[i])
        else:
            r.append(a[i])
    return q, r

def qsort(a):
    if len(a) < 2:
        return a
    if len(a) == 2:
        if a[0] > a[1]:
            return [a[1], a[0]]
        return a

    p = a[len(a) // 2]
    q, r = partition(a, p)
    q_sorted = qsort(q)
    r_sorted = qsort(r)
    return q_sorted + r_sorted

print(qsort([6, 5, 4, 3, 2, 1]))
#%%

nums = [8, 1, 2, 2, 3]
n = len(nums)
snums = nums.copy()
snums.sort()
mp = {}

# create map: value -> sorted position
# reversed - to overwrite duplicated values with lower position
for i in reversed(range(n)):
    print(snums[i], i)
    mp[snums[i]] = i

for i in range(n):
    print(mp[nums[i]])

#%%



x = 14
bx = bin(x).lstrip("0b")
print(bin(x))
print(bin(x).lstrip("0b"))
print(x)
print(bx.count("1"))
print(len(bx))


