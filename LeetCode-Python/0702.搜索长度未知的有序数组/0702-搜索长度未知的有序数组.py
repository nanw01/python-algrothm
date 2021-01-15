class Solution(object):
    def search(self, reader, target):
        """
        :type reader: ArrayReader
        :type target: int
        :rtype: int
        """
        i = 0
        tmp = reader.get(i)
        if tmp > target:
            return -1
        while tmp != 2147483647:
            if tmp == target:
                return i
            if tmp > target:
                break
            i += 1
            tmp = reader.get(i)
        return -1
