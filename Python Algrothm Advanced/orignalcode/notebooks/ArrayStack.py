class ArrayStack(object):
    def __init__(self):
        self._data = []

    def __len__(self):
        return len(self._data)

    def is_empty(self):
        return len(self._data) == 0

    # O(1)
    def push(self, e):
        self._data.append(e)

    # O(1)
    def top(self):
        if self.is_empty():
            raise ValueError('Stack is empty')
        return self._data[-1]

    # O(1)
    def pop(self):
        if self.is_empty():
            raise ValueError('Stack is empty')
        return self._data.pop()

    def printstack(self):
        for i in range(len(self._data)):
            print(self._data[i], end=" ")
        print()
