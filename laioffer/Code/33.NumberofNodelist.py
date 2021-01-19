# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def numberOfNodes(self, head):
        """
        input: ListNode head
        return: int
        """
        # write your solution here
        count = 0

        while head:
            count += 1
            head = head.next

        return count
