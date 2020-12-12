# https://www.geeksforgeeks.org/search-an-element-in-a-sorted-and-pivoted-array/



def pivotedBinarySearch(arr, key):
    n = len(arr)
    pivot = findPivot(arr, 0, n-1)
    if pivot == -1:
        return binarySearch(arr, 0, n-1, key)

    if arr[pivot] == key:
        return pivot
    if arr[0] <= key:
        return binarySearch(arr, 0, pivot-1, key)
    return binarySearch(arr, pivot+1, n-1, key)


# 先查找出范围
def findPivot(arr, left, right):

    while left + 1 < right:
        if arr[left] <= arr[right]:
            return right

        mid = left + (right-left)//2

        if arr[mid] >= arr[left]:
            left = mid + 1
        else:
            right = mid

    if arr[left] <= arr[right]:
        return right
    else:
        return left


# 进行查找
def binarySearch(arr, left, right, target):

    while left + 1 < right:
        mid = left+(right-left)//2

        if arr[mid] == target:
            right = mid
        elif arr[mid] < target:
            left = mid
        elif arr[mid] > target:
            right = mid

    if arr[left] == target:
        return left
    if arr[right] == target:
        return right

    return -1


arr1 = [5, 6, 7, 8, 9, 10, 1, 2, 3]

key = 3
print("Index of the element is : ",
      pivotedBinarySearch(arr1, key))
