class Solution:
    """
    @param nums: An integer array
    @return: The length of LIS (longest increasing subsequence)
    """

    def longestIncreasingSubsequence(self, nums):
        # write your code here
        if len(nums) <= 1:
            return len(nums)

        i, j = 0, 1
        max_len = j - i
        print(len(nums))
        while j < len(nums):
            print(i, j)
            if nums[i] < nums[j]:
                j += 1
                max_len = max(max_len, j - i)
            else:

                i = j
                j += 1

        return max_len


print(Solution().longestIncreasingSubsequence([1, 2, 3, 4, 5]))
