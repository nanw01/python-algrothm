from ArrayStack import ArrayStack
import sys

class MinStack2(ArrayStack):
    
    def __init__(self):
        super(MinStack2, self).__init__()
        self.min_stack = ArrayStack()
        
    def push(self, value):
        if value <= self.min():
            self.min_stack.push(value)
        super(MinStack2, self).push(value)
        return value
          
    def min(self):
        if self.min_stack.is_empty():
            return sys.maxsize
        else:
            return self.min_stack.top()    
      
    def pop(self):
        value = super(MinStack2, self).pop()
        if value == self.min():
            self.min_stack.pop()
        return value


minStack = MinStack2()
minStack.push(4)
minStack.push(6)
minStack.push(8)
minStack.push(3)
print(minStack.min())
minStack.pop()
minStack.pop()
print(minStack.min())