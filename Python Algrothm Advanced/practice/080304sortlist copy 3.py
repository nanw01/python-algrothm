
# 时间复杂度为nlogn
from LinkedList import Node, LinkedList


def sortList(head):
    """
    docstring
    """
    # 判断特殊情况
    if head is None or head.next is None:
        return head

    # 找到中间值
    mid = germoddle(head)
    rhead = mid.next
    mid.next = None
    # 返回合并的数值
    return mergeList(sortList(head), sortList(rhead))


def mergeList(lhead, rhead):
    """
    docstring
    """
    lstNode = head = Node(0)
    while lhead and rhead:
        if lhead.value < rhead.value:
            head.next = lhead
            lhead = lhead.next
        else:
            head.next = rhead
            rhead = rhead.next
        head = head.next
    if lhead:
        head.next = lhead
    elif rhead:
        head.next = rhead
    return lstNode.next


def germoddle(head):
    """
    docstring
    """
    if head:
        fast = slow = head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        return slow
    return head


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
