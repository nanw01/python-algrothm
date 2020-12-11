class Solution(object):
    def canConvert(self, str1, str2):
        """
        :type str1: str
        :type str2: str
        :rtype: bool
        """
        if str1 == str2:
            return True

        mapping = dict()
        for i, char in enumerate(str1):
            if str2[i] != char:#��Ҫת��
                if char in mapping: #�Ѿ��ж�Ӧ��
                    if mapping[char] != str2[i]:
                        return False
                else:
                    mapping[char] = str2[i]

        if len(set(str1)) == 26 and len(set(str2)) == 26:
            return False
        return True