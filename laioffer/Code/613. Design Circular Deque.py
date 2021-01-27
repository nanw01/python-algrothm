class Solution(object):
    def __init__(self, k):
        self._size = 0
        self._items = [None]*(k+1)
        self._first = 0
        self._last = 1

    def insertFront(self, value):
        if self.isFull():
            return False
        self._items[self._first] = value
        self._first = (self._first - 1 + len(self._items)) % len(self._items)
        self._size += 1
        return True

    def insertLast(self, value):
        if self.isFull():
            return False
        self._items[self._first] = value
        self._first = (self._last + 1) % len(self._items)
        self._size += 1
        return True

    def deleteFront(self):
        if self.isEmpty():
            return False
        self._first = (self._first + 1) % len(self._items)
        self._size -= 1
        return True

    def deleteLast(self):
        if self.isEmpty():
            return False
        self._last = (self._last - 1 + len(self._items)) % len(self._items)
        self._size -= 1
        return True

    def getFront(self):
        return -1 if self.isEmpty() else self._items[self._first + 1 % len(self._items)]

    def getRear(self):
        return -1 if self.isEmpty() else self._items[(self._last - 1 + len(self._items)) % len(self._items)]

    def isEmpty(self):
        return self._size == 0

    def isFull(self):
        return self._size == len(self._items) - 1
