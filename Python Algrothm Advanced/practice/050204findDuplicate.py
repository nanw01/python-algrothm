# ### Ex.4: Find the Duplicate Number
# Given an array nums containing n + 1 integers
# where each integer is between 1 and n (inclusive),
# prove that at least one duplicate number must exist.
# Assume that there is only one duplicate number, find the duplicate one.
# Note:
# You must not modify the array (assume the array is read only).
# You must use only constant, O(1) extra space.
# Your runtime complexity should be less than O(n2).
# There is only one duplicate number in the array, but it could be repeated more than once.


def findDuplicate(nums):

    low = 1
    high = len(nums)-1

    while low < high:
        mid = low+(high-low)//2
        count = 0
        for i in nums:
            if i <= mid:
                count += 1
        if count <= mid:
            low = mid+1
        else:
            high = mid

    return low


nums = [3, 5, 6, 3, 1, 4, 2]
print(findDuplicate(nums))
