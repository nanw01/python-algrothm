class Solution(object):
    def search(self, array, target):
        """
        input: int[] array, int target
        return: int
        """
        # write your solution here
        if len(array) == 0:
            return -1
        left, right = 0, len(array)-1
        while left+1 < right:
            mid = left+(right-left)//2

            if array[left] > array[mid]:
                if array[mid] < target and target < array[right]:
                    left = mid
                else:
                    right = mid

            else:
                if array[left] < target and target <= array[mid]:
                    right = mid
                else:
                    left = mid

        if array[left] == target:
            return left
        if array[right] == target:
            return right

        return -1


num_list = [3, 1, 1, 1, 1, 3]

s = Solution()

print(s.search(num_list, 3))
