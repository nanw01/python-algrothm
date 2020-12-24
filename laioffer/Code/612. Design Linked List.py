# Definition for singly-linked list.
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
        temp = self.head.next
        for _ in range(index):
            temp = temp.next
        return temp.val

    def addAtHead(self, val):
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        :type val: int
        :rtype: None
        """
        new_head = ListNode(val)

        new_head.next = self.head.next
        self.head.next = new_head
        self.length += 1

    def addAtTail(self, val):
        """
        Append a node of value val to the last element of the linked list.
        :type val: int
        :rtype: None
        """
        new_node = ListNode(val)
        node = self.head
        while node.next is not None:
            node = node.next
        node.next = new_node
        self.length += 1

    def addAtIndex(self, index, val):
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        :type index: int
        :type val: int
        :rtype: None
        """

        new_node = ListNode(val)
        node = self.head
        for _ in range(index):
            node = node.next
        new_node.next = node.next
        node.next = new_node
        self.length += 1

    def deleteAtIndex(self, index):
        """
        Delete the index-th node in the linked list, if the index is valid.
        :type index: int
        :rtype: None
        """

        node = self.head
        for _ in range(index):
            node = node.next
        node.next = node.next.next
        self.length -= 1


# Your Solution object will be instantiated and called as such:

obj = Solution()

# index = 0
# param_1 = obj.get(index)
val = 111
obj.addAtHead(val)
val = 222
obj.addAtTail(val)
index = 1
val = 333
obj.addAtIndex(index, val)
# index
# obj.deleteAtIndex(index)
node = obj.head.next
for _ in range(obj.length):
    print(node.val,end=' ')
    node = node.next
print()
print(obj.get(0))
print(obj.get(1))
print(obj.get(2))

