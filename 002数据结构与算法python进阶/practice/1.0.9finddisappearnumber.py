# Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.

# Find all the elements of [1, n] inclusive that do not appear in this array.

# Could you do it without extra space and in O(n) runtime? You may assume the returned list does not count as extra space.

# Example:

# Input: [4,3,2,7,8,2,3,1]

# Output: [5,6]

def findDisappearedNumbers1(nums):
    result = []
    for i in range(1, len(nums)+1):
        if(i not in nums):
            result.append(i)
    return result


def findDisappearedNumbers2(nums):
    # For each number i in nums,
    # we mark the number that i points as negative.
    # Then we filter the list, get all the indexes
    # who points to a positive number
    for i in range(len(nums)):
        index = abs(nums[i])-1
        nums[index] = -abs(nums[index])

    return [i+1 for i in range(len(nums)) if nums[i]>0]





nums = [4, 3, 2, 8, 8, 2, 3, 1]
print(findDisappearedNumbers1(nums))
print(findDisappearedNumbers2(nums))