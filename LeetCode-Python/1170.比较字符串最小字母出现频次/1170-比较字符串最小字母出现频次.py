class Solution(object):
    def numSmallerByFrequency(self, queries, words):
        """
        :type queries: List[str]
        :type words: List[str]
        :rtype: List[int]
        """
        
        def func(word):
            for char in "abcdefghijklmnopqrstuvwxyz":
                if char in word:
                    return word.count(char)
            return 0
        
        def func2(word):
            record = collections.Counter(word)
            return record[min(record.keys())]
        
        
        words_count = sorted(map(func2, words))
        queries_count = map(func2, queries)
        # print words_count, queries_count
        ans = []
        for query in queries_count:
            index = bisect.bisect(words_count, query) #bisect����Ѹ���ҳ���index���� <= query
            ans.append(len(words_count) - index)# �������ж��ٸ�����query��
        return ans
            