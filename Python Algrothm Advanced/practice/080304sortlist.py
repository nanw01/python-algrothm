
# 时间复杂度 nlogn
from LinkedList import Node, LinkedList


def sortList(head):
    if head is None or head.next is None:
        return head
    mid = getMiddle(head)
    rHead = mid.next
    mid.next = None
    return merge(sortList(head), sortList(rHead))


def merge(lHead, rHead):
    dummyNode = dummyHead = Node(0)
    while lHead and rHead:
        if lHead.value < rHead.value:
            dummyHead.next = lHead
            lHead = lHead.next
        else:
            dummyHead.next = rHead
            rHead = rHead.next
        dummyHead = dummyHead.next
    if lHead:
        dummyHead.next = lHead
    elif rHead:
        dummyHead.next = rHead
    return dummyNode.next


def getMiddle(head):
    if head is None:
        return head
    slow = head
    fast = head
    while fast.next and fast.next.next:
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
