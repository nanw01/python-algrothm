class Solution(object):
    def originalDigits(self, s):
        """
        :type s: str
        :rtype: str
        """
        # zero one   two three  four    five   six seven       eight nine 
        # z   o����2  w  r(��4)  u      f(��4)  x  s����6��    t(��3) ���e
        #����һ�����еĲ��ҵ�˳���� two, four, six, one, three, five, seven, eight, nine
        order = ["zero", "two", "four", "six", "one", "three", "five", "seven", "eight", "nine"]
        find =  ["z", "w",    "u",   "x",    "o",  "r",     "f",    "v",     "t",     "e"]
        digit = [0, 2, 4, 6, 1, 3, 5, 7, 8, 9]
        
        record = [0 for _ in range(10)]
        dic = collections.Counter(s)
        
        for idx in range(10): #��digit�����˳�����0~9
            cnt = dic[find[idx]] #����������ɼ���digit[idx]
            record[digit[idx]] += cnt #��¼����
            dic = dic - collections.Counter(order[idx] * cnt) #�ֵ����ȥ��Ӧ����ĸ
                
            if not dic:
                break
            
        ress = ""
        for i in range(10): #ת������Ŀ��Ҫ�ĸ�ʽ
            ress += str(i) * record[i]
            
        return ress
        