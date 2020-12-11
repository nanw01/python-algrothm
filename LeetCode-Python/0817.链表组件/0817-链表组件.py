# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def numComponents(self, head, G):
        """
        :type head: ListNode
        :type G: List[int]
        :rtype: int
        """
        g = []
        G = set(G)
        p = head
        while p: #�����������
            if p.val in G:
                g.append(p.val)
            p = p.next
            
        p, i, res = head, 0, 0
        seperate = True #���ڱ�ǵ�ǰ�ǲ��ǶϿ�
        while i < len(g) and p:
            if g[i] == p.val:
                if seperate == True: #��ǰ�ǶϿ�״̬
                    res += 1 #���Ը�res + 1
                    seperate = False #��ʾ��ǰ������
                p = p.next
                i += 1
            else:
                seperate = True #�Ѿ��Ͽ�
                p = p.next
        return res
        
   