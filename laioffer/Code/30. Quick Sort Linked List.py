# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def quickSort(self, head):
        """
        input: ListNode head
        return: ListNode
        """
        # write your solution here
        if head is None:
            return head

        tail = self.getTail(head)
        head, tail = self._quickSort(head, tail)
        tail.next = None

        return head

    def _quickSort(self, head, tail):
        if head is not tail:
            headLeft, tailLeft, headRef, tailRef, headRight, tailRight = self._quickSortPartition(
                head, tail)
            if headLeft is None:
                head = headRef
            else:
                headLeft, tailLeft = self._quickSort(headLeft, tailLeft)
                head = headLeft
                tailLeft.next = headRef
            if headRight is None:
                tail = tailRef
            else:
                headRight, tailRight = self._quickSort(headRight, tailRight)
                tailRef.next = headRight
                tail = tailRight
        return head, tail

    def _quickSortPartition(self, head, tail):
        reference = tail
        headRef, tailRef = reference, reference
        headLeft, tailLeft, headRight, tailRight = None, None, None, None

        sentinel = ListNode(None)
        sentinel.next = head
        node = sentinel
        while node.next is not tail:
            node = node.next
            if node.val > reference.val:
                if headRight is not None:
                    tailRight.next = node
                    tailRight = node
                else:
                    headRight = node
                    tailRight = node
            elif node.val < reference.val:
                if headLeft is not None:
                    tailLeft.next = node
                    tailLeft = node
                else:
                    headLeft = node
                    tailLeft = node
            else:
                tailRef.next = node
                tailRef = node

        return headLeft, tailLeft, headRef, tailRef, headRight, tailRight

    def getTail(self, node):
        while node.next:
            node = node.next

        return node


def printList(node):
    a = []
    while node is not None:
        a.append(node.val)
        node = node.next
    return a


def buildList(seq):
    f = ListNode(None)
    c = f
    for i in seq:
        c.next = ListNode(i)
        c = c.next

    return f.next


node = buildList([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
print(printList(node))

head = Solution().quickSort(node)
print(printList(head))
