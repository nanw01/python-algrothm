class Solution(object):

    def partition(self, lst, left, right):
        start, end = left, right-1
        from random import randrange
        rand = randrange(left, right)
        lst[rand], lst[right] = lst[right], lst[rand]

        pivot = lst[right]
        while start <= end:
            if lst[start] < pivot:
                start += 1
            elif lst[end] >= pivot:
                end -= 1
            else:
                lst[start], lst[end] = lst[end], lst[start]

        lst[start], lst[right] = lst[right], lst[start]

        return start

    def quick_sort(self, lst, left, right):
        if left >= right:
            return

        pivot = self.partition(lst, left, right)
        self.quick_sort(lst, left, pivot-1)
        self.quick_sort(lst, pivot+1, right)

    def quickSort(self, array):
        self.quick_sort(array, 0, len(array)-1)
        return array


print(Solution().quickSort(
    [1, 32, 2, 2, 34, 2, 5, 6, 7, 8, 10, 9, 3, 76, 5, 0]))
