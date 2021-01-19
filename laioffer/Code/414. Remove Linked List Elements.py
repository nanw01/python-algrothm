# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def removeElements(self, head, val):
        """
        input: ListNode head, int val
        return: ListNode
        """
        # write your solution here

        if head is None:
            return head

        dummy = pre = ListNode(None)
        pre.next = head
        while head:
            if head.val == val:
                head = head.next
                pre.next = head
            else:
                head = head.next
                pre = pre.next
        return dummy.next


l1 = ListNode(1)
l2 = ListNode(2)
l3 = ListNode(6)
l4 = ListNode(3)
l5 = ListNode(4)
l6 = ListNode(5)
l7 = ListNode(6)

l1.next = l2
l2.next = l3
l3.next = l4
l4.next = l5
l5.next = l6
l6.next = l7

node = Solution().removeElements(l1, 6)
while node:
    print(node.val, end=' ')
    node = node.next
