# ### Ex.2 Find Median from Data Stream 
# Median is the middle value in an ordered integer list. 
# If the size of the list is even, there is no middle value. 
# So the median is the mean of the two middle value.
# Examples: 
# [2,3,4] , the median is 3
# [2,3], the median is (2 + 3) / 2 = 2.5
# Design a data structure that supports the following two operations:
# void addNum(int num) - Add a integer number from the data stream to the data structure.
# double findMedian() - Return the median of all elements so far.

# ##########################

# 维护两个 heap, 根据大小进行插入,，控制两个的 size，求两个 heap 的最大跟最小

from heapq import heappush,heappushpop,heappop

class MedianFinder:

    def __init__(self):
        self.heaps = [], []

    def addNum(self, num):
        small, large = self.heaps
        heappush(small, -heappushpop(large, num))
        if len(large) < len(small):
            heappush(large, -heappop(small))

    def findMedian(self):
        small, large = self.heaps
        if len(large) > len(small):
            return float(large[0])
        return (large[0] - small[0]) / 2.0



finder = MedianFinder()
finder.addNum(2)
finder.addNum(3)
finder.addNum(4)
finder.findMedian()