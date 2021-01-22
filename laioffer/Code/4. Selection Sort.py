class Solution(object):
    def solve(self, array):
        """
        input: int[] array
        return: int[]
        """
        # write your solution here
        for i in range(len(array)):
            min = i
            for j in range(i+1, len(array)):
                if array[j] < array[min]:
                    min = j
            array[i], array[min] = array[min], array[i]
        return array

    def select_sort(self, lst):

        for i in range(len(lst)):
            min = i
            for j in range(i+1, len(lst)):
                if lst[j] < lst[min]:
                    min = j
            lst[i], lst[min] = lst[min], lst[i]

        return lst


print(Solution().select_sort([3, 5, 7, 8, 6, 4, 4, 3, 2, 2]))
