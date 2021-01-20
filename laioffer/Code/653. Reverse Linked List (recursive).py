# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def reverse(self, head):
        """
        input: ListNode head
        return: ListNode
        """
        # write your solution here
        node = head
        if (node == None):
            return node
        if (node.next == None):
            return node
        node1 = self.reverse(node.next)
        node.next.next = node
        node.next = None

        return node1
