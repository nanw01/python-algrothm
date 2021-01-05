class Solution(object):
    def localMinimum(self, array):
        """
        input: int[] array
        return: int
        """
        # write your solution here
        return self.lowerFinder(array, 0, len(array)-1)

    def lowerFinder(self, array, start, end):
        if start == end:
            return start

        if start+1 == end:
            if array[start] < array[end]:
                return start
            else:
                return end

        mid = start+(end-start)//2
        if array[mid] < array[mid-1] and array[mid] < array[mid+1]:
            return mid

        if array[mid] < array[mid-1] and array[mid] > array[mid+1]:
            return self.lowerFinder(array, mid+1, end)
        else:
            return self.lowerFinder(array, start, mid-1)
