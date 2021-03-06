# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def isPalindrome(self, head):
        """
        input: ListNode head
        return: boolean
        """
        # write your solution here
        if head is None or head.next is None:
            return True

        slow = head
        fast = head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        prev = None
        while slow:
            newNode = slow.next
            slow.next = prev
            prev = slow
            slow = newNode

        while prev:
            if prev.val != head.next:
                return False
            prev = prev.next
            head = head.next

        return True
