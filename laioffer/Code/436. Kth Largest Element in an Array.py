class Solution(object):
    def findKthLargest(self, nums, k):
        """
        input: int[] nums, int k
        return: int
        """
        # write your solution here
        import heapq
        lst = []

        for n in nums:
            heapq.heappush(lst, n)
            if len(lst) > k:
                heapq.heappop(lst)
        return heapq.heappop(lst)


s = Solution()

lst = [3, 2, 1, 5, 6, 4]
print(s.findKthLargest(lst, 2))
