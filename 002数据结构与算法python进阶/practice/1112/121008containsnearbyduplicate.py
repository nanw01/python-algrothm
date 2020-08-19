# ### Ex.8 Contains Duplicates II

# Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the difference between i and j is at most k.


def containsNearbyDuplicate(nums, k):
    dic = {}
    for i, v in enumerate(nums):
        if v in dic and i - dic[v] <= k:
            return True
        dic[v] = i
    return False


nums = [1,2,3,4,5]
print(containsNearbyDuplicate(nums, 1))
print(containsNearbyDuplicate(nums, 6))
nums = [1,2,3,4,5,3]
print(containsNearbyDuplicate(nums, 1))
print(containsNearbyDuplicate(nums, 2))
print(containsNearbyDuplicate(nums, 3))