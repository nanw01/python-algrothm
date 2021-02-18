# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def addTwoNumbers(self, node1, node2):
        """
        input: ListNode l1, ListNode l2
        return: ListNode
        """
        # write your solution here

        newNode1 = node1
        newNode2 = node2
        fakeNode = ListNode(None)
        curr = fakeNode
        carry = 0
        while newNode1 and newNode2:
            tempSum = newNode1.val + newNode2.val + carry
            carry = tempSum // 10
            curr.next = ListNode(tempSum % 10)
            curr = curr.next
            newNode1 = newNode1.next
            newNode2 = newNode2.next

        while newNode1:
            tempSum = newNode1.val + carry
            carry = tempSum // 10
            curr.next = ListNode(tempSum % 10)
            curr = curr.next
            newNode1 = newNode1.next

        while newNode2:
            tempSum = newNode2.val + carry
            carry = tempSum // 10
            curr.next = ListNode(tempSum % 10)
            curr = curr.next
            newNode2 = newNode2.next

        if carry > 0:
            curr.next = ListNode(carry)

        return fakeNode.next
