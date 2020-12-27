# 使用列表实现堆栈

class ArrayStack(object):

    def __init__(self):
        self._data = []

    def __len__(self):
        return len(self._data)

    def is_empty(self):
        return len(self._data) == 0

    def push(self, v):
        self._data.append(v)

    def top(self):
        if self.is_empty():
            raise ValueError('Stack is empty')
        return self._data[-1]

    def pop(self):
        if self.is_empty():
            raise ValueError('Stack is empty')
        return self._data.pop()

    def printstack(self):
        for i in range(len(self._data)):
            print(self._data[i], end=" ")
        print()


mystack = ArrayStack()
print('size was: ', str(len(mystack)))
mystack.push(1)
mystack.push(2)
mystack.push(3)
mystack.push(4)
mystack.push(5)
print('size was: ', str(len(mystack)))
mystack.printstack()
mystack.pop()
mystack.pop()
print('size was: ', str(len(mystack)))
mystack.printstack()
print(mystack.top())
mystack.pop()
mystack.pop()
mystack.pop()
# mystack.pop()
