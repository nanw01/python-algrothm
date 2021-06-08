# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        head = None
        temp = None
        c = 0
        while l1 or l2:
            if not l1:
                a = 0
            else:
                a = l1.val
            if not l2:
                b = 0
            else:
                b = l2.val
            n = a + b + c
            c = 1 if n > 9 else 0
            node = ListNode(n % 10)
            if not head:
                head = node
                temp = node
            else:
                head.next = node
                head = node
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        if c:
            node = ListNode(1)
            head.next = node
        return temp
