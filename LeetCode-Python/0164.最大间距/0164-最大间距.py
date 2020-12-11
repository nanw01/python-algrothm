class Solution(object):
    def maximumGap(self, nums):
        if len(nums) < 2:
            return 0
        min_val, max_val = min(nums), max(nums)
        if min_val == max_val:
            return 0
        
        n = len(nums) + 1 # Ͱ�ĸ���
        step = (max_val - min_val) // n
        
        exist = [0 for _ in range(n + 1)]  #��ʾͰ�Ƿ�Ϊ��
        max_num = [0 for _ in range(n + 1)]#��ʾͰ��Ԫ�ص����ֵ
        min_num = [0 for _ in range(n + 1)]#��ʾͰ��Ԫ�ص���Сֵ
        
        for num in nums: #�����е�����Ͱ
            idx = self.findBucketIndex(num, min_val, max_val, n) 
            max_num[idx] = num if not exist[idx] else max(num, max_num[idx])
            min_num[idx] = num if not exist[idx] else min(num, min_num[idx])
            exist[idx] = 1
        res = 0
        pre = max_num[0]
        for i in range(1, n + 1):
            if exist[i]:
                res = max(res, min_num[i] - pre)
                pre = max_num[i]
        return res
                        
    def findBucketIndex(self, num, min_val, max_val, n):
        return int((num - min_val) * n / (max_val - min_val))
                