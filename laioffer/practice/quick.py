class Solution:

    def partition(self, lst, left, right):
        start, end = left, right-1
        import random

        rand = random.randrange(left, right)
        lst[rand], lst[right] = lst[right], lst[rand]
        pivot = lst[right]
        while start <= end:
            if lst[start] < pivot:
                start += 1
            elif lst[end] >= pivot:
                end -= 1
            else:
                lst[start], lst[end] = lst[end], lst[start]
                start += 1
                end -= 1
            lst[start], lst[right] = lst[right], lst[start]

        return start

    def quick_sort_helper(self, lst, left, right):
        if left >= right:
            return
        pivot = self.partition(lst, left, right)
        self.quick_sort_helper(lst, left, pivot-1)
        self.quick_sort_helper(lst, pivot+1, right)

    def quick_sort(self, lst):
        self.quick_sort_helper(lst, 0, len(lst)-1)
        return lst


print(Solution().quick_sort(
    [2, 1, 3, 4, 6, 87, 5, 4, 3, 12, 2, 3, 4, 5, 6, 7, 9, 0]))
