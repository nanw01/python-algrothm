from LinkedList import Node
# 时间复杂度 nlogn


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