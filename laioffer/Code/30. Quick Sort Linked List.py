# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def quickSort(self, head):
        """
        input: ListNode head
        return: ListNode
        """
        # write your solution here
        if head is None or head.next is None:
            return head

        pivot = head.val
