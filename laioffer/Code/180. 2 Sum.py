class Solution(object):
    def existSum(self, array, target):
        """
        input: int[] array, int target
        return: boolean
        """
        # write your solution here

        array = sorted(array)
        print(array)
        left, right = 0, len(array) - 1

        while left < right:

            sum = array[left] + array[right]

            if sum == target:
                return True
            elif sum < target:
                left += 1
            else:
                right -= 1

        return False


s = Solution()

print(s.existSum([3, 2, 4], 5))
