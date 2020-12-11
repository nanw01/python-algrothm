class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if not nums:
            return []

        window = []
        res = []
        for i in range(len(nums)):
            if window and window[0] <= i - k: #��ǰwindowͷӦ�ñ�����
                window.pop(0)
                
            while window and nums[window[-1]] < nums[i]: #�� nums[i] С�Ķ���Ҫ����ΪֻҪ���ڵ����ֵ
                window.pop()
                
            window.append(i)
            if i >= k - 1:
                res.append(nums[window[0]])
        return res