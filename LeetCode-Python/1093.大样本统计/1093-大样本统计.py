class Solution(object):
    def sampleStats(self, count):
        """
        :type count: List[int]
        :rtype: List[float]
        """
        s = 0
        total_cnt = sum(count)
        cnt  = 0
        avg, median, mean, mean_cnt = 0, 0, 0, 0
        min_element, max_element = 0, 0
        #����Сֵ
        for i in range(len(count)):
            if count[i] != 0:
                min_element = i
                break
        #�����ֵ
        for i in range(len(count) - 1, -1, -1):
            if count[i] != 0:
                max_element = i
                break
                
        #��һ��ͳ���˶��ٸ�����        
        geshu = 0
        for i in count:
            if i > 0:
                geshu += i
                
        find = False
        for i, num in enumerate(count):
            s += num * i #�����ܺ�
            if mean_cnt < num: #��count�������ֵ���±�
                mean = i
                mean_cnt = num
                
            cnt += num #��Ŀǰ�����˶��ٸ�����
            if cnt > total_cnt // 2 and find == False: 
                if total_cnt % 2: #��λ���϶���һ����
                    median = i
                    find = True
                else:
                    if cnt - num == total_cnt // 2: #��λ����������ͬ����
                        for j in range(i - 1, -1, -1): #��ǰ����һ����
                            if count[j] > 0:
                                median = (i + j) /2.0
                                find = True
                                break
                    else:#��λ����������ͬ����
                        median = i
                        find = True
                                
        return [min_element, max_element, 1.0 * s /geshu, median, mean ]
                                
                                
            