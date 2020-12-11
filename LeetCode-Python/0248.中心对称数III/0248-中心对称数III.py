class Solution(object):
    def strobogrammaticInRange(self, low, high):
        """
        :type low: str
        :type high: str
        :rtype: int
        """
        if int(low) > int(high):
            return 0
        self.findStrobogrammatic(len(high))
        
        low_rec = self.record[len(low)]
        #����low_rec���ж��ٸ��� >= low���������low_cnt��
        #�����൱���������߽�

        low_cnt = len(low_rec) - bisect.bisect_left(low_rec, low)
        
        #�ҵ�һ�� > high�������±꣬���û�ҵ�����˵��������������е������� highС        
        high_rec = self.record[len(high)]
        high_cnt = bisect.bisect_right(high_rec, high)

        if len(low) + 1 == len(high):
            return low_cnt + high_cnt
        elif len(low) == len(high):
            return low_cnt + high_cnt - len(high_rec)
        else:
            tmp = 0
            for l in range(len(low) + 1, len(high)):
                tmp += len(self.record[l])
            return tmp + low_cnt + high_cnt

        
        
 
    def findStrobogrammatic(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        self.record = dict()
        self.record[0] = ["0"]
        self.record[1] = ["0", "1", "8"]
        self.record[2] = ["11", "69", "88", "96"]
        pair = ["00", "11", "88", "69", "96"]
        if n <= 2:
            return self.record[n]
        cnt = 3
        while cnt <= n:
            tmp = []
            if (cnt - 1) % 2 == 0: #���ǰһ����ż�����ȣ���ôֱ�����м�ӳ���Ϊ1�ľͿ���
                for item in self.record[cnt - 1]:
                    for num in self.record[1]:
                        tmp.append(item[:len(item)// 2] + num + item[len(item) // 2:])
            else:                  #���ǰһ�����������ȣ���ô�����м�ӳ���Ϊ2�ľͿ��� ��ע��Ҫ����ӡ�00��
                for item in self.record[cnt - 2]:
                    for num in pair:
                        tmp.append(item[:len(item)// 2] + num + item[len(item) // 2:])
            self.record[cnt] = sorted(tmp, key = lambda x: int(x))
            cnt += 1