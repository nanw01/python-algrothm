class Solution(object):
    def closest(self, array, target):
        """
        input: int[] array, int target
        return: Integer[]
        """
        # write your solution here

        left, right = 0, len(array) - 1
        while left + 1 < right:
            mid = left + (right - left) // 2

            if array[mid] < target:
                left = mid
            elif array[mid] > target:
                right = mid
            else:
                return [array[0], array[mid]]

        #

        return [array[0], array[right]]


s = Solution()
print(s.closest([1, 4, 7, 13], 7))
