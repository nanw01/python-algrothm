class Solution:
    from random import randrange

    def partition(self, lst, start, end, pivot_index):
        lst[pivot_index], lst[end] = lst[end], lst[pivot_index]

        store_index = start
        pivot = lst[end]

        for i in range(start, end):
            if lst[i] < pivot:
                lst[i], lst[store_index] = lst[store_index], lst[i]
                store_index += 1

        lst[store_index], lst[end] = lst[end], lst[store_index]

        return store_index

    def quick_sort_helper(self, lst, start, end):
        if start >= end:
            return
        pivot_index = self.randrange(start, end+1)
        new_pivot_index = self.partition(lst, start, end, pivot_index)
        self.quick_sort_helper(lst, start, new_pivot_index-1)
        self.quick_sort_helper(lst, new_pivot_index+1, end)\


    def quickSort(self, lst):
        self.quick_sort_helper(lst, 0, len(lst)-1)
        return lst


print(Solution().quickSort([2, 1, 3, 6, 4, 5, 9, 7, 8, 0]))
