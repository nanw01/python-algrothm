# Find the sum of contiguous subarray within a one-dimensional array of numbers which has the largest sum.

import sys










# O(n) 动态规划
def subarray3(alist):
    result = -sys.maxsize
    local = 0
    for i in alist:
        local = max(local + i, i)
        result = max(result, local)
    return result



alist = [-2,-3,4,-1,-2,1,5,-3]
print(subarray3(alist))