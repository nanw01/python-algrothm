class Solution(object):
    def smallerPairs(self, array, target):
        """
        input: int[] array, int target
        return: int
        """
        # write your solution here
        array = sorted(array)

        count = 0
        res = []
        for i in range(len(array)):
            for j in range(i+1, len(array)):
                temp_sum = array[i]+array[j]
                if temp_sum < target:
                    count += 1
        return count


s = Solution()
print(s.smallerPairs([-1, 0, 1], 2))
