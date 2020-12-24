class Solution(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.fake_head = ListNode(None)
        self.size = 0

    def get(self, index):
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        :type index: int
        :rtype: int
        """
        temp = self.fake_head.next
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

        new_head.next = self.fake_head.next
        self.fake_head.next = new_head
        self.size += 1

    def addAtTail(self, val):
        """
        Append a node of value val to the last element of the linked list.
        :type val: int
        :rtype: None
        """
        self.get(self.size).next = ListNode(val)
        self.size += 1

    def addAtIndex(self, index, val):
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        :type index: int
        :type val: int
        :rtype: None
        """
        new_node = ListNode(val)

        new_node.next = self.get(index+1)
        self.get(index).next = new_node
        self.size += 1

    def deleteAtIndex(self, index):
        """
        Delete the index-th node in the linked list, if the index is valid.
        :type index: int
        :rtype: None
        """
        if index == 0:
            pass

        self.get(index-1).next = self.get(index).next
        self.size -= 1

# Your Solution object will be instantiated and called as such:
obj = Solution()
param_1 = obj.get(index)
obj.addAtHead(val)
obj.addAtTail(val)
obj.addAtIndex(index,val)
obj.deleteAtIndex(index)
