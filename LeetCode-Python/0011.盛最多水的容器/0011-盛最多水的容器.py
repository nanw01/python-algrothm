class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left, right = 0, len(height) - 1
        res = 0
        while(left < right):
            # print left, right,
            res = max(res, (right - left) * min(height[left], height[right]))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return res


class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        res = 0
        while left < right:
            res = max(res, (right-left) * min(height[left], height[right]))
            if height[left] > height[right]:
                right -= 1
            else:
                left += 1

        return res
