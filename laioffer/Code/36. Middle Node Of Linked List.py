# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def middleNode(self, head):
        """
        input: ListNode head
        return: ListNode
        """
        # write your solution here
        if head is None or head.next is None or head.next.next is None:
            return head

        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        return slow


node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)
node4 = ListNode(4)
node5 = ListNode(5)

node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node5


s = Solution()
print(s.middleNode(node1))
