class Solution(object):
    def maxSumAfterPartitioning(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        dp = [0 for _ in range(len(A))]
        
        for i, x in enumerate(A): #ɨ��ÿ����
            subarray_max = x
            for j in range(1, K + 1): # J ����ǰ������ĳ��ȣ�����A[i], ��������A[i - (j - 1): i + 1]
                if i - (j - 1) >= 0:#������������ǰ�����ⳤ��Ϊ J �������飬
                    subarray_max = max(subarray_max, A[i - (j - 1)]) #ȷ��subarray_max�Ǵ�����������ֵ
                    #��ôдsubarray_max = max(A[i - (j - 1): i + 1]]Ҳ���Թ������Ǻ���
                    
                    if i - j < 0:  # A[i]֮ǰǡ����j - 1��Ԫ�أ�������һ������˳���ΪJ�������飬�൱�ڵ�ǰ��������Ա�ʾΪA[:j]
                        dp[i] = max(dp[i], subarray_max * j)
                    else:
                        dp[i] = max(dp[i], dp[i - j] + subarray_max * j)

        return dp[-1]