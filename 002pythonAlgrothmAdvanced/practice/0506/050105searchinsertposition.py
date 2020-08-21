# Given a sorted array and a target value,
# return the index if the target is found.
# If not, return the index where it would be if it were inserted in order.
# You may assume no duplicates in the array


def search_insert_position(alist, target):
    if len(alist) == 0:
        return 0

    left, right = 0, len(alist)-1

    while left+1 < right:

        mid = left+(right-left)//2

        if alist[mid] == target:
            return mid
        if alist[mid] < target:
            left = mid
        else:
            right = mid

    if alist[left] >= target:
        return left
    if alist[right] >= target:
        return right

    return right+1


num_list = [5, 7, 9]
print(search_insert_position(num_list, 10))
