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
        n1 = self._reverseList(l1)
        n2 = self._reverseList(l2)

        fake_head = cur_node = ListNode(None)
        carry = 0
        while n1 and n2:

            tem_sum = n1.val + n2.val + carry
            carry = tem_sum // 10
            cur_node.next = ListNode(tem_sum % 10)
            cur_node = cur_node.next
            n1, n2 = n1.next, n2.next
        while n1:
            tem_sum = n1.val + carry
            carry = tem_sum // 10
            cur_node.next = ListNode(tem_sum % 10)
            cur_node = cur_node.next
            n1 = n1.next
        while n2:
            tem_sum = n2.val + carry
            carry = tem_sum // 10
            cur_node.next = ListNode(tem_sum % 10)
            cur_node = cur_node.next
            n2 = n2.next
        if carry > 0:
            cur_node.next = ListNode(carry)

        return self._reverseList(fake_head.next)

    def _reverseList(self, head):

        if head == None or head.next == None:
            return head
        p = self._reverseList(head.next)
        head.next.next, head.next = head, None
        return p
