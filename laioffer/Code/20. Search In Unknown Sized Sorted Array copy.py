# Definition for a unknown sized dictionary.
# class Dictionary(object):
#   def get(self, index):
#     pass

class Solution(object):
    def search(self, dic, target):
        """
        input: Dictionary dic, int target
        return: int
        """
        # write your solution here
        start = 1
        while dic.get(start) and dic.get(start) < target:
            start = start * 2

        left, right = 0, start

        while left + 1 < right:
            mid = (left+right) // 2
            if dic.get(mid) is None or dic.get(mid) > target:
                right = mid
            elif dic.get(mid) < target:
                left = mid
            else:
                return map

        if dic.get(left) == target:
            return left
        if dic.get(right) == target:
            return right

        return -1
