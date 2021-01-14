# 1.0.7  Ex.7 Search in Sorted Array with Empty Strings
# Given a sorted array of strings which is interspersed with empty strings,
# write a meth­od to find the location of a given string.


def search_empty(alist, target):
    if len(alist) == 0:
        return -1

    left, right = 0, len(alist)-1

    while left+1 < right:

        while left+1 < right and alist[right] == '':
            right -= 1
        if alist[right] == '':
            right -= 1
        if left > right:
            return -1

        mid = left+(right-left)//2

        while alist[mid]=='':
            mid+=1


        if alist[mid] == target:
            return mid
        if alist[mid] < target:
            left = mid + 1
        else:
            right = mid-1

    if alist[left] == target:
        return left
    if alist[right] == target:
        return right

    return -1


arr = ["for", "geeks", "", "", "", "", "ide",
       "practice", "", "", "", "quiz"]
print(search_empty(arr, 'geeks'))
