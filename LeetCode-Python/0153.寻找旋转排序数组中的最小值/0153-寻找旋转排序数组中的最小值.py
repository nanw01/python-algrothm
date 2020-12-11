class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        if nums[0] < nums[-1]: #û������ת
            return nums[0]
        
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            
            if mid + 1< len(nums): #mid �������һλ
                if nums[mid - 1] > nums[mid] and nums[mid] < nums[mid + 1]: #�ҵ���
                    return nums[mid]
            else:
                if nums[mid - 1] > nums[mid]: # mid �����һλ
                    return nums[mid]
            if nums[mid] < nums[-1]: 
                right = mid - 1
            else:
                left = mid + 1
                
        
                    
        