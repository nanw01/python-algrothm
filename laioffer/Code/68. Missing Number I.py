class Solution(object):
    def missing(self, array):
        """
        input: int[] array
        return: int
        """
        # write your solution here
        return list(set(list(range(1, len(array)+2)))-set(array))[0]


s = Solution()
print(s.missing([]))
