class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        s = set(nums)
        for num in range(len(nums)):
            print(num)
            if num+1 not in s:
                return num+1
        return len(nums)


nums = [2, 1, 5, 4]
print(set(nums))


s = Solution()
print(s.missingNumber(nums))
