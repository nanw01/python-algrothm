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


def twoSum2(num, target):
    index = []
    numtosort = num[:]
    numtosort.sort()
    i = 0
    j = len(numtosort) - 1
    while i < j:
        if numtosort[i] + numtosort[j] == target:
            for k in range(0, len(num)):
                if num[k] == numtosort[i]:
                    index.append(k)
                    break
            for k in range(len(num)-1, -1, -1):
                if num[k] == numtosort[j]:
                    index.append(k)
                    break
            index.sort()
            break
        elif numtosort[i] + numtosort[j] < target:
            i = i + 1
        elif numtosort[i] + numtosort[j] > target:
            j = j - 1

    return (index[0]+1, index[1]+1)


print(twoSum2([1, 2, 3, 4, 5, 6, 7], 0))
