class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.s = []
        self.min_s = []

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.s.append(x)
        if self.min_s:
            self.min_s.append(min(x, self.min_s[-1]))
        else:
            self.min_s.append(x)

    def pop(self):
        """
        :rtype: None
        """
        if len(self.s) == 0:
            return -1
        self.min_s.pop()
        return self.s.pop()

    def top(self):
        """
        :rtype: int
        """
        if len(self.s) == 0:
            return -1
        return self.s[-1]

    def getMin(self):
        """
        :rtype: int
        """
        if len(self.s) == 0:
            return -1
        return self.min_s[-1]


# Your MinStack object will be instantiated and called as such:


obj = MinStack()

obj.push(1)
obj.push(3)
obj.push(5)
obj.pop()
param_3 = obj.top()
param_4 = obj.getMin()


print(param_3)
print(param_4)
