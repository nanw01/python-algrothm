class Solution(object):
    def binarySearch(self, array, target):
        """
        input: int[] array, int target
        return: int
        """
        # write your solution here
        if len(array) == 0:
            return -1

        left, right = 0, len(array)-1
        while left+1 < right:
            mid = (left+right)//2

            if array[mid] < target:
                left = mid
            else:
                right = mid

        if array[left] == target:
            return left
        if array[right] == target:
            return right

        return -1
