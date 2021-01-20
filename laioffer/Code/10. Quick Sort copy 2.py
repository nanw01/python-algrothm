class Solution(object):
    def quickSort(self, array):
        """
        input: int[] array
        return: int[]
        """
        # write your solution here
        self.quick_sort(array, 0, len(array)-1)
        return array

    def quick_sort(self, array, left, right):
        if left >= right:
            return

        pivot = self.partition(array, left, right)
        self.quick_sort(array, left, pivot-1)
        self.quick_sort(array, pivot+1, right)

    def partition(self, alist, left, right):
        start, end = left, right-1
        import random
        rand = random.randint(left, right)
        alist[rand], alist[end] = alist[end], alist[rand]
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


print(Solution().quickSort([4, 7, 32, 4, 8, 56, 898, 6, 4, 456, 32, 6, 5437, 4, 7856, 8, 5,
                            567, 56, 45, 3, 2, 54, 325, 54, 6745, 765, 867, 34, 879, 870, 7, 975, 46, 5]))
