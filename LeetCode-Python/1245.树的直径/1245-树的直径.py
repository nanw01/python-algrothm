from collections import defaultdict        
class Solution(object):
    def treeDiameter(self, edges):
        """
        :type edges: List[List[int]]
        :rtype: int
        """
        if not edges:
            return 0

        self.neibors = defaultdict(set)
        self.res = 0
        
        for start, end in edges: # ����
            self.neibors[start].add(end)
        
        def getHeight(node):
            res = []
            for neibor in self.neibors[node]:
                res.append(getHeight(neibor))
            
            while len(res) < 2: # ������������������͸������յ���ȥ
                res.append(0)
                
            res = sorted(res)
            self.res = max(self.res, sum(res[-2:])) # ȡ���������������
            return 1 + max(res)
        
        getHeight(edges[0][0])
        return self.res

        