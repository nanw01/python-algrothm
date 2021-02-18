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

        newNode1 = self.reverse1(node1)
        newNode2 = self.reverse1(node2)
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

        return self.reverse1(fakeNode.next)

    def reverse1(self, head):
        prevNode = None
        while head:
            nextNode = head.next
            head.next = prevNode
            prevNode = head
            head = nextNode

        return prevNode


def buildList(seq):
    f = ListNode(None)
    c = f
    for i in seq:
        c.next = ListNode(i)
        c = c.next

    return f.next


def printList(node):
    a = []
    while node is not None:
        a.append(node.val)
        node = node.next
    return a


node1 = buildList([1, 2, 3])
node2 = buildList([1, 7, 8, 9])
nodes = Solution().addTwoNumbers(node1, node2)
print(printList(nodes))
