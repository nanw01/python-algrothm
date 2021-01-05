# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def mergeSort(self, head):
        """
        input: ListNode head
        return: ListNode
        """
        # write your solution here
        if head is None or head.next is None:
            return head

        mid = self.getMiddle(head)
        midnext = mid.next

        mid.next = None

        left = self.mergeSort(head)
        right = self.mergeSort(midnext)

        sortedLinkedList = self.sortList(left, right)
        return sortedLinkedList

    def sortList(self, a, b):
        """
        docstring
        """
        result = None

        if a == None:
            return b
        if b == None:
            return a

        if a.val < b.val:
            result = a
            result.next = self.sortList(a.next, b)
        else:
            result = b
            result.next = self.sortList(a, b.next)
        return result

    def getMiddle(self, head):

        if head is None:
            return head

        slow = fast = head
        while fast.next is not None and fast.next.next is not None:
            slow = slow.next
            fast = fast.next.next

        return slow


def printList(head):
    if head is None:
        print(' ')
        return
    curr_node = head
    while curr_node:
        print(curr_node.val, end=" ")
        curr_node = curr_node.next
    print(' ')


if __name__ == "__main__":
    s = Solution()

    node1 = ListNode(1)
    node2 = ListNode(3)
    node3 = ListNode(5)
    node4 = ListNode(7)
    node5 = ListNode(9)
    node6 = ListNode(2)
    node7 = ListNode(4)
    node8 = ListNode(6)

    node1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node5
    node5.next = node6
    node6.next = node7
    node7.next = node8

    new = s.mergeSort(node1)

    printList(new)
