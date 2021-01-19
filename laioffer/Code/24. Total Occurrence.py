class Solution(object):
    def totalOccurrence(self, array, target):
        """
        input: int[] array, int target
        return: int
        """
        # write your solution here
        if len(array) == 0:
            return 0
        first = self.findStart(array, target)
        end = self.lastStart(array, target)
        print(first, end)
        return 0 if end == -1 else end-first+1

    def findStart(self, array, target):
        if len(array) == 0:
            return 0
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

    def lastStart(self, array, target):
        if len(array) == 0:
            return 0
        left, right = 0, len(array)-1
        while left+1 < right:
            mid = (left+right)//2

            if array[mid] > target:
                right = mid
            else:
                left = mid
        if array[right] == target:
            return right

        if array[left] == target:
            return left
        return -1


print(Solution().totalOccurrence([1, 2, 2, 2, 5, 8], 2))
