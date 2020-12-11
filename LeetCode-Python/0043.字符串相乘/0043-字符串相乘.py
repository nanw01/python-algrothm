class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        if num1 == "0" or num2 == "0": #�����������
            return "0"
        
        l1, l2 = len(num1), len(num2) 
        if l1 < l2: 
            num1, num2 = num2, num1 #����num1ʼ�ձ�num2��
            l1, l2 = l2, l1
            
        num2 = num2[::-1]
        res = "0"
        for i, digit in enumerate(num2):
            tmp = self.StringMultiplyDigit(num1, int(digit)) + "0" * i #����num1��num2�ĵ�ǰλ�ĳ˻�
            res = self.StringPlusString(res, tmp) #����res��tmp�ĺ�

        return res
    
    def StringMultiplyDigit(self,string, n):
        #��������Ĺ����ǣ�����һ���ַ�����һ�������ĳ˻��������ַ���
        #����������Ϊ "123", 3�� ����"369"
        s = string[::-1]
        res = []
        for i, char in enumerate(s):
            num = int(char)
            res.append(num * n)
        res = self.CarrySolver(res)
        res = res[::-1]
        return "".join(str(x) for x in res)
        
    def CarrySolver(self, nums):  
        #��������Ĺ����ǣ�������������е�ÿһλ����ý�λ
        #����������[15, 27, 12], ����[5, 8, 4, 1]
        i = 0
        while i < len(nums):
            if nums[i] >= 10:
                carrier = nums[i] // 10
                if i == len(nums) - 1:
                    nums.append(carrier)
                else:
                    nums[i + 1] += carrier
                nums[i] %= 10
            i += 1
                    
        return nums
    
    def StringPlusString(self, s1, s2):
        #��������Ĺ����ǣ����������ַ����ĺ�
        #����������Ϊ��123���� ��456��, ����Ϊ"579"
        l1, l2 = len(s1), len(s2)
        if l1 < l2:
            s1, s2 = s2, s1
            l1, l2 = l2, l1
        s1 = [int(x) for x in s1]
        s2 = [int(x) for x in s2]
        s1, s2 = s1[::-1], s2[::-1]
        for i, digit in enumerate(s2):
            s1[i] += s2[i]
            
        s1 = self.CarrySolver(s1)
        s1 = s1[::-1]
        return "".join(str(x) for x in s1)
            
        
        
            
            