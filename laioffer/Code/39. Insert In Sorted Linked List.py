# 39. Insert In Sorted Linked List
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def insert(self, head, value):
        """
        input: ListNode head, int value
        return: ListNode
        """
        # write your solution here
        if head is None:
            return ListNode(value)
        dummy = pre = ListNode(0)
        dummy.next = head
        while head:
            if value <= head.val:
                newNode = ListNode(value)
                pre.next = newNode
                newNode.next = head
                return dummy.next
            elif value > head.val and head.next is None:
                newNode = ListNode(value)
                head.next = newNode
                pre = pre.next
                return dummy.next

            head = head.next
            pre = pre.next


l0 = ListNode(0)
l1 = ListNode(1)
l3 = ListNode(3)

l0.next = l1
# l1.next = l3

res = Solution().insert(l0, 2)
while res:
    print(res.val, end=' ')
    res = res.next
