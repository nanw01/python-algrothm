from LinkedList import Node, LinkedList


class StackWithQueue:

    def __init__(self):
        self.queue = LinkedList()

    # Push element x onto stack.
    def push(self, x):
        self.queue.add_last(x)

    # Removes the element on top of the stack.
    def pop(self):
        size = self.queue.size()
        for _ in range(1, size):
            self.queue.add_last(self.queue.remove_first())
        self.queue.remove_first()

    def top(self):
        size = self.queue.size()
        for _ in range(1, size):
            self.queue.add_last(self.queue.remove_first())
        result = self.queue.remove_first()
        self.queue.add_last(result)
        return result


stack = StackWithQueue()
stack.push(1)
stack.push(2)
print(stack.top())


stack = StackWithQueue()
stack.push(1)
stack.push(2)
stack.pop()
stack.push(3)
print(stack.top())
