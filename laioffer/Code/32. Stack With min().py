class Solution(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.lst = []
        self.minn = []

    def push(self, x):
        """
        input : int x
        return : 
        """
        self.lst.append(x)
        if len(self.minn) != 0:
            self.minn.append(min(x, self.minn[-1]))
        else:
            self.minn.append(x)

    def pop(self):
        """
        return : int
        """
        if len(self.lst) == 0:
            return - 1

        self.minn.pop()
        return self.lst.pop()

    def top(self):
        """
        return : int
        """
        if len(self.lst) == 0:
            return - 1
        return self.lst[-1]

    def min(self):
        """
        return : int
        """
        if len(self.minn) == 0:
            return -1
        return self.minn[-1]


obj = Solution()

obj.push(5)
print(obj.top())
print(obj.min())
print(obj.pop())
print(obj.min())
print(obj.top())
