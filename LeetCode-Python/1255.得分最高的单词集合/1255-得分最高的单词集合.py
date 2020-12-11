class Solution(object):
    def maxScoreWords(self, words, letters, score):
        """
        :type words: List[str]
        :type letters: List[str]
        :type score: List[int]
        :rtype: int
        """
        from collections import defaultdict, Counter
        dic = dict()
        letter_dic = defaultdict(int)
        for i, val in enumerate(score):#����һ���ֵ�
            dic[chr(ord("a") + i)] = val #key�� ��ĸ��val����ĸ��Ӧ�ķ���
            
        letter_dic = Counter(letters)#������һ���ֵ䣬 key����ĸ�� val��ÿ����ĸʣ��ĸ���
 
        s = set(letters)
        v_words = []
        for word in words:#ɾ�����и��������ܱ����ɵĵ���
            flag = 0
            for char in word:
                if char not in s:
                    flag = 1
            if flag: # ���һ�����������ĳ����letters���Ҳ�������ĸ�������迼���������
                continue
            v_words.append(word)
        self.res = 0
                
        def helper(word, letter_dic):
            # return True ���word����letter_dic���letter���ɣ����򷵻�False
            dicc = collections.Counter(word)
            for key in dicc:
                if dicc[key] > letter_dic[key]:
                    return False
            return True
        
        def dfs(start, tmp):
            self.res = max(self.res, tmp)
            if start >= len(v_words):
                return
            
            for i in range(start, len(v_words)):#��start��ʼ�ң������ظ�
                if helper(v_words[i], letter_dic):#�����ǰ���ʿ��Ա�����
                    for char in v_words[i]: #�������������ֵ�
                        letter_dic[char] -= 1
                    dfs(i + 1, tmp + sum([dic[char] for char in v_words[i]])) #dfs��һ��
                    for char in v_words[i]: #���ݣ���ԭ����״̬
                        letter_dic[char] += 1                   
        dfs(0, 0)
        return self.res