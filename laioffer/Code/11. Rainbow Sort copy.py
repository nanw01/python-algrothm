class Solution(object):
    def rainbowSort(self, array):
        """
        input: int[] array
        return: int[]
        """
        # write your solution here
        left, right = 0, len(array)-1
        index = 0
        while index <= right:
            if array[index] == -1:
                array[index], array[left] = array[left], array[index]
                left += 1
                index += 1
            elif array[index] == 1:
                array[index], array[right] = array[right], array[index]
                right -= 1
            else:
                index += 1

        return array


nums = [1, 1, 0, -1, 0, 1, -1]

print(Solution().rainbowSort(nums))
