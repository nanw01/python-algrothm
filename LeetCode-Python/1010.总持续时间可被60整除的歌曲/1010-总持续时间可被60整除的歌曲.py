class Solution(object):
    def numPairsDivisibleBy60(self, time):
        """
        :type time: List[int]
        :rtype: int
        """
        record = [0 for _ in range(0, 60)]
        for index, item in enumerate(time):
            record[item % 60] += 1

        res = 0
        for i in range(0, 60):
            if i in [0, 30] and record[i] > 1:
                res += record[i] * (record[i] - 1) # ����N�����Ա�60����������˵����������ȡ�����н���ĸ���ΪC N 2 = N *(N - 1) / 2
                record[i] = 0 # һ�δ��������еĿ��Ա�60����������Ȼ���record[0]���㣬��֤���ظ�����
            elif i:            
                res += record[60 - i] * record[i]

        return res // 2