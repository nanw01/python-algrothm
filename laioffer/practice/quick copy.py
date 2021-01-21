class Solution(object):
    def partition(self, alist, left, right):
        start = left
        end = right-1
        import random
        rand = random.randrange(left, right)
        alist[rand], alist[right] = alist[right], alist[rand]
        pivot = alist[right]
        while start <= end:
            if alist[start] < pivot:
                start += 1
            elif alist[end] >= pivot:
                end -= 1
            else:
                alist[start], alist[end] = alist[end], alist[start]
                start += 1
                end -= 1

        alist[start], alist[right] = alist[right], alist[start]

        return start

    def quick_sort(self, alist, left, right):
        if left >= right:
            return

        pivot = self.partition(alist, left, right)
        self.quick_sort(alist, left, pivot-1)
        self.quick_sort(alist, pivot+1, right)

    def quickSort(self, array):
        self.quick_sort(array, 0, len(array)-1)
        return array


print(Solution().quickSort(
    array=[2, 1, 3, 8, 7, 6, 5, 0, 9, 4, 5, 6, 7, 4, 3, 2, 2, 7]))
