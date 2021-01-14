class Solution(object):
    def allTriples(self, array, target):
        """
        input: int[] array, int target
        return: int[][]
        """
        # write your solution here
        result = []
        for i in array[:-2]:
            m = target-i
            ss = set()
            for j in array[1:-1]:
                n = m-j
                if n in ss:
                    result.append([i, j, n])
                else:
                    ss.add(j)
        return result


print(Solution().allTriples([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3))
