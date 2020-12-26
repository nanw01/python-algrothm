
# 时间复杂度为nlogn
from LinkedList import Node, LinkedList


def sortList(head):
    if head is None or head.next is None:
        return head
    mid = getMiddle(head)
    rHead = mid.next
    mid.next = None
    return merge(sortList(head), sortList(rHead))


def merge(left, right):
    node = head = Node(0)
    while left and right:
        if left.value < right.value:
            head.next = left
            left = left.next
        else:
            head.next = right
            right = right.next
        head = head.next
    if left:
        head.next = left
    else:
        head.next = right
    return node.next


def getMiddle(head):
    if head is None:
        return head
    slow = fast = head

    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next
    return slow


node1 = Node(9)
node2 = Node(1)
node3 = Node(13)
node4 = Node(6)
node5 = Node(5)
node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node5
node = sortList(node1)
lst = LinkedList()
lst.head.next = node
lst.printlist()
