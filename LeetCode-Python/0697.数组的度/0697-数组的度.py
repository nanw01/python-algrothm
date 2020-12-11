class Solution(object):
    def findShortestSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 1. ���������degree
        # 2. �ҵ���ЩԪ�س�����degree��
        # 3. ����2��Ԫ�ص�һ�γ��ֵ��±�����һ�γ��ֵ��±��������Ӧ������ĳ���
        degree = 0
        for digit in set(nums):
            degree = max(degree, nums.count(digit))
            
        candidates = list()
        for digit in set(nums):
            if nums.count(digit) == degree:
                candidates.append(digit)
                
        l = len(nums)
        reversenums = nums[::-1]
        res = l
        for candidate in candidates:
            left_pos = nums.index(candidate)
            right_pos = l - reversenums.index(candidate)

            res = min(res, right_pos - left_pos)
            
        return res