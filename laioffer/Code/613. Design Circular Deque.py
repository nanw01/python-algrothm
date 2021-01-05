class Solution(object):

    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the deque to be k.
        :type k: int
        """
        self.queue = []
        self.size = k
        self.front = 0
        self.rear = 0

    def insertFront(self, value):
        """
        Adds an item at the front of Deque. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        if not self.isFull():
            self.queue.insert(0, value)
            self.rear += 1
            return True
        else:
            return False

    def insertLast(self, value):
        """
        Adds an item at the rear of Deque. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        if not self.isFull():
            self.queue.append(value)
            self.rear += 1
            return True
        else:
            return False

    def deleteFront(self):
        """
        Deletes an item from the front of Deque. Return true if the operation is successful.
        :rtype: bool
        """

    def deleteLast(self):
        """
        Deletes an item from the rear of Deque. Return true if the operation is successful.
        :rtype: bool
        """

    def getFront(self):
        """
        Get the front item from the deque.
        :rtype: int
        """

    def getRear(self):
        """
        Get the last item from the deque.
        :rtype: int
        """

    def isEmpty(self):
        """
        Checks whether the circular deque is empty or not.
        :rtype: bool
        """

    def isFull(self):
        """
        Checks whether the circular deque is full or not.
        :rtype: bool
        """


# Your Solution object will be instantiated and called as such:
# obj = Solution(3)
# param_1 = obj.insertFront(1)
# param_2 = obj.insertLast(2)
# param_3 = obj.deleteFront()
# param_4 = obj.deleteLast()
# param_5 = obj.getFront()
# param_6 = obj.getRear()
# param_7 = obj.isEmpty()
# param_8 = obj.isFull()

# print(param_1)
# print(param_2)
# print(param_3)
# print(param_4)
# print(param_5)
# print(param_6)
# print(param_7)
# print(param_8)
