class Solution(object):
    def shiftPosition(self, array):
        """
        input: int[] array
        return: int
        """
        # write your solution here
        if len(array) == 0:
            return - 1

        l, r = 0, len(array)-1

        while l + 1 < r:
            mid = l + (r - l) // 2

            if array[l] < array[mid]:
                if array[mid] > array[r]:
                    l = mid
                else:
                    r = mid
            else:
                if array[mid] > array[r]:
                    l = mid
                else:
                    r = mid

        return l if array[l] < array[r] else r


s = Solution()
print(
    s.shiftPosition([1, 2, 3, 4, 5])
)
