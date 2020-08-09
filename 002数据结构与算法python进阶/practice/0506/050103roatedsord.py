# 1.0.3  Ex.3 Find Min in Rotated Sorted Array
# Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. Find the minimum element.

def search(lst):
    if len(lst) == 0:
        return -1

    left = 0
    right = len(lst)-1

    while left + 1 < right:

        if lst[left]<lst[right]:
            return lst[left]

        mid = left+(right-left)//2

        if left < mid:
            left = mid
        elif left > mid: 
            right = mid

    return lst[left] if lst[left] < lst[right] else lst[right]


num_list = [10,1,2,3,5,7,8,9]
print(search(num_list))