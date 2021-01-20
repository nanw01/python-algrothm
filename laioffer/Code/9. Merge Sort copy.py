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
        lefthalf = self.mergeSort(array=array[:mid])
        righthalf = self.mergeSort(array=array[mid:])

        return self.merge(lefthalf, righthalf)

    def merge(self, lefthalf, righthalf):
        i, j = 0, 0
        alist = []

        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist.append(lefthalf[i])
                i += 1
            else:
                alist.append(righthalf[j])
                j += 1

        while i < len(lefthalf):
            alist.append(lefthalf[i])
            i += 1

        while j < len(righthalf):
            alist.append(righthalf[j])
            j += 1

        return alist


print(Solution().mergeSort([4, 7, 32, 4, 8, 56, 898, 6, 4, 456, 32, 6, 5437, 4, 7856, 8, 5,
                            567, 56, 45, 3, 2, 54, 325, 54, 6745, 765, 867, 34, 879, 870, 7, 975, 46, 5]))
