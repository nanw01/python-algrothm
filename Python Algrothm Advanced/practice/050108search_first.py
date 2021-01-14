# 1.0.8  Ex.8 Search 1st Position of element in Infinite Array¶
# 阶梯思路，，，，两倍的 R
def search_first(alist):
    left, right = 0, 1

    while alist[right] == 0:
        left = right
        right *= 2
        if right > len(alist):
            right = len(alist)-1
            break

    print(left, right)
    alist = alist[0:right+1]

    print(alist)

    left, right = 0, len(alist)-1

    while left+1 < right:

        mid = left+(right-left)//2

        if alist[mid] == 1:
            return mid

        if alist[mid] < 1:
            left = mid+1

        else:
            right = mid

    if alist[left] == 1:
        return left
    if alist[right] == 1:
        return right

    return -1


alist = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
print(search_first(alist))
