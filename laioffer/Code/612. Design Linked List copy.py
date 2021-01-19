class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.head = ListNode(None)
        self.length = 0

    def get(self, index):
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        :type index: int
        :rtype: int
        """
        if index > self.length:
            return -1
        res = self.head
        for _ in range(index+1):
            res = res.next
        return res.val

    def addAtHead(self, val):
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        :type val: int
        :rtype: None
        """
        newNode = ListNode(val)
        newNode.next = self.head.next
        self.head.next = newNode
        self.length += 1

    def addAtTail(self, val):
        """
        Append a node of value val to the last element of the linked list.
        :type val: int
        :rtype: None
        """
        tail = self.head
        for _ in range(self.length):
            tail = tail.next

        newTail = ListNode(val)
        tail.next = newTail
        self.length += 1

    def addAtIndex(self, index, val):
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        :type index: int
        :type val: int
        :rtype: None
        """
        prev = self.head
        for _ in range(index):
            prev = prev.next
        newNode = ListNode(val)
        newNode.next = prev.next
        prev.next = newNode
        self.length += 1

    def deleteAtIndex(self, index):
        """
        Delete the index-th node in the linked list, if the index is valid.
        :type index: int
        :rtype: None
        """
        prev = self.head
        for _ in range(index-1):
            prev = prev.next
        prev.next = prev.next.next
        self.length -= 1


# Your Solution object will be instantiated and called as such:
# obj = Solution()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index, val)
# obj.deleteAtIndex(index)

obj = Solution()

# index = 0
# param_1 = obj.get(index)

obj.addAtHead(1)
obj.addAtTail(3)
obj.addAtIndex(1, 2)
print('get', obj.get(1))
obj.deleteAtIndex(1)
print('get', obj.get(1))

node = obj.head.next
for _ in range(obj.length):
    print(node.val, end=' ')
    node = node.next

# val = 222
# obj.addAtTail(val)
# index = 1
# val = 333
# obj.addAtIndex(index, val)
# # index
# # obj.deleteAtIndex(index)
# node = obj.head.next
# for _ in range(obj.length):
#     print(node.val, end=' ')
#     node = node.next
# print()
# print(obj.length)

# # print(obj.get(0))
# print(obj.getNode(1))
# # print(obj.get(2))

# print("atTail", obj.addAtTail(989))
# print("atTail", obj.addAtTail(989))
# print("atTail", obj.addAtTail(989))

# node = obj.head
# for i in range(obj.length+1):
#     print(i, node.val)
#     node = node.next
