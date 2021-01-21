class Solution(object):
    def isOneEditDistance(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        edit = 0 #�����༭����
        distance = len(s) - len(t)
        if abs(distance) > 1: #���ȶ����˲�ֹһλ���϶�����
            return False
        if not s or not t:
            return s != t
        i, j = 0, 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]: #����༭
                i += 1 #����ָ�붼˳�����һλ
                j += 1
            else:
                if edit: #Ψһ�Ļ����Ѿ�������
                    return False
                if distance == 1: #ɾ��
                    i += 1
                elif distance == -1:#����
                    j += 1
                else: #�滻
                    i += 1
                    j += 1
                edit += 1
        # print i, j, edit
        if i < len(s):
            return edit == 0
        if j < len(t):
            return edit == 0
        return i == len(s) and j == len(t) and edit == 1
    