class Solution(object):
    def closest(self, array, target):
        """
        input: int[] array, int target
        return: Integer[]
        """
        # write your solution here
        array.sort()
        left, right = 0, len(array) - 1
        max_diff = float('inf')
        res = [array[left], array[right]]
        while left < right:

            sum_val = array[left] + array[right]
            if abs(target - sum_val) < max_diff:
                max_diff = abs(target - sum_val)
                res = [array[left], array[right]]

            if sum_val < target:
                left += 1
            else:
                right -= 1

        return res


s = Solution()
print(s.closest([1, 4, 6, 13], 7))
