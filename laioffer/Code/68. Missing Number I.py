class Solution(object):
    def missing(self, array):
        """
        input: int[] array
        return: int
        """
        # write your solution here
        return set(list(range(1, len(array)+1)))-set(array)


s = Solution()
print(s.missing([12, 11, 10, 9, 4, 5, 6, 7, 2, 3, 8]))
