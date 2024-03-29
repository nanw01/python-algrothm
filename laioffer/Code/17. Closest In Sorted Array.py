class Solution(object):
    def closest(self, array, target):
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
            elif array[mid] > target:
                right = mid
            else:
                return mid

        return left if abs(array[left]-target) < abs(array[right]-target) else right


s = Solution()
a = [1, 4, 6]
b = 3
print(s.closest(a, b))
