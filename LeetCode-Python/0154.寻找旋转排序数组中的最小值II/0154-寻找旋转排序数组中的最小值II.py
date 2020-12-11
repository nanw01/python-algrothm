class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            
            if nums[mid] > nums[right]: #��Сֵ�϶��ڡ�mid + 1, right��
                left = mid + 1
            elif nums[mid] < nums[right]: #��Сֵ�϶��ڡ�left, mid��
                right = mid
            elif nums[mid] == nums[right]: #�޷�ȷ����Сֵ������
                
                flag = False
                for j in range(right - 1, mid, -1):#������Ҳ��ң����ǲ����б�nums[mid]��С����
                    if nums[mid] > nums[j]:
                        flag = True #ȷʵ���ڸ�С��������nums[j]
                        break
                        
                if flag: #����ȷ����Сֵ�϶��ڡ�mid + 1, j��
                    left = mid + 1
                    right = j
                else: #mid ��right���е��������, ��Сֵ�϶��ڡ�left, mid��
                    right = mid
                    
        return nums[left]
                    