# Find in Rotated Array
def search(alist, target):
    if len(alist) == 0:
        return -1    
    left, right = 0, len(alist) - 1
    while left + 1 < right: 
        mid = left + (right - left) // 2
        if alist[mid] == target:
            return mid
        
        if (alist[left] < alist[mid]):
            if alist[left] <= target and target <= alist[mid]:
                right = mid
            else:
                left = mid
        else:
            if alist[mid] <= target and target <= alist[right]:
                left = mid
            else: 
                right = mid
                            
    if alist[left] == target:
        return left
    if alist[right] == target:
        return right
        
    return -1


num_list = [10, 22, 33, 5, 7, 8, 9]
print(search(num_list,22))
