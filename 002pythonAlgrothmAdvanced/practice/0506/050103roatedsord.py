# 1.0.3  Ex.3 Find Min in Rotated Sorted Array
# Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. Find the minimum element.


def search(alist):
    if len(alist) == 0:
        return -1

    left, right = 0, len(alist) - 1

    while left + 1 < right:
        if alist[left] < alist[right]:
            return alist[left]

        mid = left + (right - left) // 2

        if alist[mid] >= alist[left]:
            left = mid + 1
        else:
            right = mid

    return alist[left] if alist[left] < alist[right] else alist[right]


num_list = [10, 22, 33, 5, 7, 8, 9]
print(search(num_list))
