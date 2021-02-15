# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        input: ListNode l1, ListNode l2
        return: ListNode
        """
        # write your solution here
        dummy = ListNode(0)
        pre = dummy
        carry = 0
        while l1 or l2 or carry:

            # node1 = ListNode(2)
            # node2 = ListNode(4)
            # node3 = ListNode(3)

            # node1.next = node2
            # node2.next = node3

            # node9 = ListNode(5)
            # node8 = ListNode(6)
            # node7 = ListNode(4)


            # node9.next = node8
            # node8.next = node7
node1 = ListNode(9)
node2 = ListNode(8)

node9 = ListNode(1)
node8 = ListNode(1)


node = Solution().addTwoNumbers(node1, node9)

while node:
    print(node.val)
    node = node.next
