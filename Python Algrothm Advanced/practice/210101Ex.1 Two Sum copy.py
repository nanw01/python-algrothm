# Ex.1 Two Sum
# Given an array of integers, find two numbers such that they add up to a specific target number.

class Solution(object):
    def existSum(self, array, target):
        """
        input: int[] array, int target
        return: boolean
        """
        # write your solution here

        s = set()
        for n in array:
            req = target - n
            if req in s:
                return True
            else:
                s.add(n)

        return False


def twoSum(nums, k):
    s = set()
    for n in nums:
        req = k - n
        if req in s:
            return True
        else:
            s.add(n)

    return False


def twoSum1(nums, target):
    dic = {}
    for i, num in enumerate(nums):
        if num in dic:
            return [dic[num], i]
        else:
            dic[target - num] = i


print(twoSum([1, 2, 3, 4, 5, 6, 7], 0))

# print(enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9]))
for i, num in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9]):
    print(i, num)

print(twoSum1([1, 2, 3, 4, 5, 6, 7, 8, 9], 3))
