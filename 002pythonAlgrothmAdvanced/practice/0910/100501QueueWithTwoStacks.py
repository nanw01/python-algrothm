# 1.0.1  Ex.1 Implement Queue using Stacks
# 使用堆栈实现一个队列


class QueueWithTwoStacks:
    
    def __init__(self):
        self.insertStack = []
        self.popStack = []

    def enqueue(self, e):
        self.insertStack.append(e)
        return e
    
    def dequeue(self):
        if len(self.insertStack)==0 and len(self.popStack)==0:
            return None
        
        if len(self.popStack)==0:
            while len(self.insertStack)!=0:
                self.popStack.append(self.insertStack.pop())
        
        return self.popStack.pop()



mystack = QueueWithTwoStacks()
e = mystack.enqueue(3)
print(e)
e = mystack.enqueue(2)
print(e)
e = mystack.enqueue(1)
print(e)
e = mystack.dequeue()
print(e)
e = mystack.dequeue()
print(e)