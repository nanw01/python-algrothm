# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        newhead = ListNode(-1)
        newhead.next = head

        cnt = 1
        slow = head
        while cnt < m - 1:
            slow = slow.next
            cnt += 1
        cnt = 1
        fast = head
        while cnt < n:
            # print fast.val, n
            fast = fast.next
            cnt += 1
            # print fast.val, cnt
        tail = fast.next
        fast.next = None
        if m != 1:
            slow.next = self.reverseLL(slow.next)
        else:
            newhead.next = self.reverseLL(slow)
        p = slow
        while p and p.next:
            p = p.next
        p.next = tail
        return newhead.next

    def reverseLL(self, head):
        if not head or not head.next:
            return head

        p = self.reverseLL(head.next)
        head.next.next = head
        head.next = None
        return p
