from collections import deque
class Solution(object):
    def minMutation(self, start, end, bank):
        bank = set(bank)
        if end not in bank:
            return -1
        q = deque()
        q.append([start, 0])
        visited = set()
        char = "ACGT"
        while q: #��ʼBFS
            cur, cnt = q.popleft() #ȡһ������
            if cur == end: #����ҵ���
                return cnt
            
            for i in range(len(cur)):
                for j in range(4):
                    new = cur[:i] + char[j] + cur[i + 1:] #���ɴӵ�ǰ������Ա任�����»���
                    
                    if new in bank and new not in visited: #����»�����Ч
                        visited.add(new)
                        q.append([new, cnt + 1])
                        
        return -1
            