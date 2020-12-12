class Solution(object):
    def searchInsert(self, input, target):
        """
        input: int[] input, int target
        return: int
        """
        # write your solution here

        if len(input) < 1:
            return 0

        left, right = 0, len(input)

        while left < right:
            mid = left+(right-left)//2

            if input[mid] < target:
                left = mid+1
            elif input[mid] > target:
                right = mid-1
            else:
                if input[left]==input[mid]:
                    right = left
                elif input[left]<input[mid]:
                    right = mid

        return left


s = Solution()

a = [1,3,3,3,5,6]
b = 3

print(s.searchInsert(a, b))
