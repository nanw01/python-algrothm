class Solution(object):
    def quickSort(self, array):
        """
        input: int[] array
        return: int[]
        """
        # write your solution here
        if len(array) <= 1:
            return array

        left = self.quickSort([x for x in array[1:] if x <= array[0]])
        right = self.quickSort([y for y in array[1:] if y > array[0]])

        return left + [array[0]]+right


print(Solution().quickSort([4, 7, 32, 4, 8, 56, 898, 6, 4, 456, 32, 6, 5437, 4, 7856, 8, 5,
                            567, 56, 45, 3, 2, 54, 325, 54, 6745, 765, 867, 34, 879, 870, 7, 975, 46, 5]))
