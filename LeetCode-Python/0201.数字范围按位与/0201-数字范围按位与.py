class Solution(object):
    def rangeBitwiseAnd(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        if m == 0 or m == n:
            return m
        else: #��n > m��ʱ����Ϊ���������� &�Ľ�������һλ�ض�Ϊ1����˿��Եݹ鴦��
            return self.rangeBitwiseAnd(m >> 1, n >> 1) << 1
        
            