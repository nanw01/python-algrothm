from heapq import *
class DinnerPlates(object):
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.stack = []
        self.c = capacity
        self.idx = [] #���ڴ���п�λ��ջ���±�

    def push(self, val):
        """
        :type val: int
        :rtype: None
        """
        if self.idx:
            index = heappop(self.idx) #����С�Ŀյ�ջ���±�
            self.stack[index].append(val) #����val
            if len(self.stack[index]) < self.c: #�������֮��û�����Ͱ�����յ�ջ���±�Ž�self.idx
                heappush(self.idx, index)
        else: #��������ջ�����ˣ�ֻ������һ��ջ��ĩβ
            self.stack.append([val])
            if self.c > 1:
                self.idx.append(len(self.stack) - 1)
            

    def pop(self):
        """
        :rtype: int
        """
        while self.stack and not self.stack[-1]:
            self.stack.pop()
        if not self.stack: #���е�ջ���ǿյ�
            return -1
        else:
            if len(self.stack[-1]) == self.c: #�������������ջ
                heappush(self.idx, len(self.stack) - 1) #����Ҫ����п�λ��ջ��
            return self.stack[-1].pop()
            
    def popAtStack(self, index):
        """
        :type index: int
        :rtype: int
        """
        if index >= len(self.stack): #�±�Խ��
            return -1
        else:
            s = self.stack[index] # ���±�Ϊindex��ջȡ����
            if len(s) == self.c: #������ջ����������
                heappush(self.idx, index) #����Ҫ�����
            return s.pop() if s else -1 #������ջ�ǿյģ�����-1�� ���򷵻�pop


# Your DinnerPlates object will be instantiated and called as such:
# obj = DinnerPlates(capacity)
# obj.push(val)
# param_2 = obj.pop()
# param_3 = obj.popAtStack(index)