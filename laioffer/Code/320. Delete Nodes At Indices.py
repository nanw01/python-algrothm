# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def deleteNodes(self, head, indices):
        """
        input: ListNode head, int[] indices
        return: ListNode
        """
        # write your solution here

        for i in range(len(indices)):
            head = self.deleteNode(head, indices[i]-i)

        return head

    def deleteNode(self, head, index):
        """
        input: ListNode head, int index
        return: ListNode
        """
        # write your solution here
        if head is None:
            return head

        fakeNode = ListNode(None)
        fakeNode.next = head
        prev = fakeNode
        for i in range(index):
            if head.next is None:
                return fakeNode.next

            prev = prev.next
            head = head.next

        prev.next = head.next

        return fakeNode.next
