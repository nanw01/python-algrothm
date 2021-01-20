class Solution(object):
    def mergeSort(self, array):
        """
        input: int[] array
        return: int[]
        """
        # write your solution here

        if len(array) <= 1:
            return array

        mid = len(array)//2
        l = self.mergeSort(array[:mid])
        r = self.mergeSort(array[mid:])

        c = []

        while len(l) > 0 and len(r) > 0:
            if l[0] < r[0]:
                c.append(l[0])
                l.remove(l[0])
            else:
                c.append(r[0])
                r.remove(r[0])

        if len(l) == 0:
            c += r
        else:
            c += l

        return c


print(Solution().mergeSort([4, 7, 32, 4, 8, 56, 898, 6, 4, 456, 32, 6, 5437, 4, 7856, 8, 5,
                            567, 56, 45, 3, 2, 54, 325, 54, 6745, 765, 867, 34, 879, 870, 7, 975, 46, 5]))
