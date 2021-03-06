class Solution(object):
    def allTriples(self, array, target):
        """
        input: int[] array, int target
        return: int[][]
        """
        # write your solution here
        dd = set()
        ret = []
        for i in range(0, len(array)-1):
            for j in range(i+1, len(array)):
                cur = target - array[i] - array[j]
                if cur in dd:
                    ret.append([cur, array[i], array[j]])
                else:
                    dd.add(i)

        return ret


print(Solution().allTriples([1, 1, 1, 1, 1, 1], 3))
