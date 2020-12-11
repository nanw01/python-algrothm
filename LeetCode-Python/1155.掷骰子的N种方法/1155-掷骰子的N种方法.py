class Solution(object):
    def numRollsToTarget(self, d, f, target):
        """
        :type d: int
        :type f: int
        :type target: int
        :rtype: int
        """
        record = dict()
        def backtrack(dice, face, t):
            if (dice, face, t) in record: #����Ѿ�֪����ǰ����Ľ�
                return record[(dice, face, t)] #ֱ�Ӷ�ȡ����
            
            if dice == 0: #��û�����ӵ�ʱ���ж�һ���ǲ��Ǹպ��ҵ�target
                return 1 if t == 0 else 0
            
            if t < 0 or dice <= 0: #��Ч��ʱ��û�н⣬���Է���0
                return 0
            tmp = 0 #tmp���ڼ�¼��ǰ�����ж��ٸ���
            for i in range(1, face + 1): #�������е����
                tmp += backtrack(dice - 1, face, t - i) 
                
            record[(dice, face, t)] = tmp #�ѽ�������������Ժ���
            return tmp
        
        backtrack(d, f, target)
        return max(record.values()) % (10 ** 9 + 7) #���Ľ����ĸ����Ľ�
                
            