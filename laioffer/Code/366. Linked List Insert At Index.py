# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def insert(self, head, index, value):
        """
        input: ListNode head, int index, int value
        return: ListNode
        """
        # write your solution here
        # if head is None and index == 0:
        #   return ListNode(value)

        dummy = prev = ListNode(0)
        prev.next = head
        i = 0
        while prev:
            if index == i:
                newNode = ListNode(value)
                prev.next = newNode
                if head:
                    newNode.next = head
                return dummy.next
            i += 1
            prev = prev.next
            if head:
                head = head.next
        return dummy.next


l0 = ListNode(0)
l1 = ListNode(1)
l3 = ListNode(3)

l0.next = l1
# l1.next = l3

res = Solution().insert(None, 1, 20)
while res:
    print(res.val, end=' ')
    res = res.next
