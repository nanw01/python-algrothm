class Solution(object):
    def existSum(self, a, b, target):
        """
        input: int[] a, int[] b, int target
        return: boolean
        """
        # write your solution here
        for i in range(len(a)):
            temp = target-a[i]
            if temp in set(b):
                return True

        return False


s = Solution()
print(s.existSum([3, 4, -1, 0], [5, -1, 2], -3))
