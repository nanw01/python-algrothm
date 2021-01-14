# 返回去重列表的长度

def removeDuplicates(alist):
    if not alist:
        return 0

    tail = 0

    for j in range(1, len(alist)):
        if alist[j] != alist[tail]:
            tail += 1
            alist[tail] = alist[j]

    return tail + 1


def removeDuplicates1(alist):

    i = 0
    for n in alist:
        if i == 0 or alist[i - 1] < n:
            alist[i] = n
            i += 1
    return i


alist = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
alist1 = [1, 1, 2]
print(removeDuplicates(alist))
print(removeDuplicates1(alist1))
