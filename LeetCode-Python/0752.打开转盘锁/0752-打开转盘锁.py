from collections import deque
class Solution(object):
    def openLock(self, deadends, target):
        """
        :type deadends: List[str]
        :type target: str
        :rtype: int
        """
        deadends = set(deadends)
        if "0000" in deadends: #�������㶼�����߾�88
            return -1
        
        queue = deque()
        queue.append(["0000", 0])
        cnt = 0

        while queue:
            node, cnt = queue.popleft() #ȡһ���������cnt�ǵ�ǰ�ߵĲ���
            if node == target: #�ҵ���
                return cnt     

            for i in range(4):
                for j in [1, -1]:
                    next_node = node[:i] + str((int(node[i]) + j) % 10) + node[i + 1:] 

                    if next_node not in deadends: #�µĵ�����߶���û�߹�
                        deadends.add(next_node) #�����ظ�
                        queue.append([next_node, cnt + 1])

        return -1