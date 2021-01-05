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
