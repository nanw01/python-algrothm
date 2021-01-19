class Solution(object):
    def sqrt(self, x):

        if x == 0:
            return 0

        left, right = 1, x
        while True:
            mid = (left+right)//2

            if mid*mid > x:
                right = mid
            elif mid**2 <= x and (mid+1)**2 > x:
                return mid
            else:
                left = mid+1
