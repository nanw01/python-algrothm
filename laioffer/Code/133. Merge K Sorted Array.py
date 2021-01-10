class Solution(object):
    def merge(self, arrayOfArrays):
        """
        input: int[][] arrayOfArrays
        return: int[]
        """
        # write your solution here
        from heapq import merge
        res = []
        for arr in arrayOfArrays:
            res = list(merge(res, arr))
        return res


s = Solution()

lst = [[3, 2, 1, 5, 6, 4], [0, -2, -9, -4, -7, -8]]
print(s.merge(lst))
