
def search_range(alist, target):

    if len(alist) == 0:
        return (-1, -1)

    lbound, rbound = 0, len(alist)-1
    # 先找左边边界

    left, right = 0, len(alist)-1

    while left+1 < right:
        mid = left+(right-left)//2
        if alist[mid] < target:
            left = mid
        elif alist[mid] > target:
            right = mid
        else:
            right = mid

    if alist[left] == target:
        lbound = left
    elif alist[right] == target:
        lbound = right
    else:
        return (-1, -1)

    # 后找有边界
    left, right = 0, len(alist)-1

    while left+1 < right:
        mid = left+(right-left)//2
        if alist[mid] < target:
            left = mid
        elif alist[mid] > target:
            right = mid
        else:
            left = mid

    if alist[left] == target:
        rbound = left
    elif alist[right] == target:
        rbound = right
    else:
        return (-1, -1)

    return (lbound, rbound)


num_list = [5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9]
print(search_range(num_list, 7))
