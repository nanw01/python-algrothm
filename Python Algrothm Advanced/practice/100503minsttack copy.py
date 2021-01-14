# Design a stack that supports push, pop, top,
# and retrieving the minimum element in constant time.
# getMin() -- Retrieve the minimum element in the stack.


from ArrayStack import ArrayStack
import sys


class MinStack(ArrayStack):
    def __init__(self):
        super(MinStack, self).__init__()

    def push(self, v):
        newMin = min(v, self.min())
        super(MinStack, self).push(NodeWithMin(v, newMin))

    def min(self):
        if super(MinStack, self).is_empty():
            return sys.maxsize
        else:
            return super(MinStack, self).top()._min


class NodeWithMin(object):
    def __init__(self, v, min):
        self._value = v
        self._min = min


minStack = MinStack()
minStack.push(4)
minStack.push(6)
minStack.push(8)
minStack.push(3)
print(minStack.min())
minStack.pop()
minStack.pop()
print(minStack.min())
