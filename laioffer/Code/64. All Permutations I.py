class Solution(object):
    def permutations(self, input):
        """
        input: string input
        return: string[]
        """
        # write your solution here
        return self._perm('', input, [])

    def _perm(self, result, nums, s):

        if (len(nums) == 0):
            s.append(result)

        for i in range(len(nums)):
            self._perm(result+str(nums[i]), nums[0:i]+nums[i+1:], s)  #

        return s


print(Solution().permutations('abc'))
