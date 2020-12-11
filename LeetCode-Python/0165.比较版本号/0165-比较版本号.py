class Solution(object):
    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        l1 = version1.split(".")
        l2 = version2.split(".")
        i, j = 0, 0
        while i < len(l1) and j < len(l2): # ��λ�ȴ�С
            num1, num2 = int(l1[i]), int(l2[j])
            if num1 < num2:
                return -1
            elif num1 > num2:
                return 1
            i += 1
            j += 1

        if len(l1) == len(l2) and num1 == num2: #������ȶ������һλҲ���
            return 0

        if len(l1) > len(l2): #���l1���ڶ�l2������
            while i < len(l1) and int(l1[i]) == 0:  #�ж�l1�����ǲ���ȫ��0
                i += 1
            if i == len(l1): #���ȫ��0�������
                return 0
            else:
                return 1
            
        if len(l2) > len(l1):
            while j < len(l2) and int(l2[j]) == 0: #�ж�l2�����ǲ���ȫ��0
                j += 1
            if j == len(l2): #���ȫ��0�������
                return 0
            else:
                return -1
            return -1