class Solution(object):
    def lengthOfLongestSubstringTwoDistinct(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) <= 2:
            return len(s)
        start, end = 0, 0
        q = []
        res = 0
        dic = {} #��¼q����ĸ���һ�γ��ֵ��±�
        for i, char in enumerate(s):
            dic[char] = i
            if len(q) < 2:
                if char not in q:
                    q.append(char)
            else:
                if char not in q: #Ҫ�Ծɻ�����
                    if dic[q[0]] < dic[q[1]]:
                        tmp = q[0]
                        q.pop(0)
                    else:
                        tmp = q[1]
                        q.pop(1)
                    start = dic[tmp] + 1
                    
                    q.append(char)
            end = i   
            res = max(end - start + 1, res)
        return res
                    