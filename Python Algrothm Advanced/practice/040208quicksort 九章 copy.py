# quick sort
class Solution:
    # @param {int[]} A an integer array
    # @return nothing

    def sortIntegers(self, A):

        # Write your code here
        self.quickSort(A, 0, len(A) - 1)

    def quickSort(self, A, start, end):
        if start >= end:
            return

        left, right = start, end
        pivot = A[(left+right)//2]
        while left <= right:

            while left <= right and A[left] < pivot:
                left += 1
            while left <= right and A[right] > pivot:
                right -= 1
            if left <= right:
                A[left], A[right] = A[right], A[left]
                left += 1
                right -= 1
        print(start, right, left, end)
        self.quickSort(A, start, right)
        self.quickSort(A, left, end)


l = [34, 54, 75, 8, 55, 38, 2, 56, 7, 84, 96, 98, 76, 12, 71, 86, 19, 36, 52]
Solution().sortIntegers(l)
print(l)
