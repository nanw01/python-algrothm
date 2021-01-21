class Solution:

    def quickSortHelper(self, lst, left, right):
        """
        docstring
        """
        from random import randrange
        if len(lst) <= 1:
            return lst

        pivot_index = randrange(left, right)
        new_pivot_index = self.partition(lst, left, right, pivot_index)

    def quickSort(self, lst):
        """
        docstring
        """
        self.quickSortHelper(lst, 0, len(lst)-1)


print(Solution().quickSort([2, 1, 3, 6, 4, 5, 9, 7, 8, 0]))
