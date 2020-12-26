from LinkedList import Node, LinkedList


def reverseBetween(head, m, n):
    if m == n:
        return head

    dummyNode = Node(0)
    dummyNode.next = head
    pre = dummyNode

    for _ in range(m - 1):
        pre = pre.next

    # reverse the [m, n] nodes
    result = None
    current = pre.next
    for _ in range(n - m + 1):
        nxt = current.next
        current.next = result
        result = current
        current = nxt

    pre.next.next = current
    pre.next = result

    return dummyNode.next



lst = LinkedList()
lst.add_last(1)
lst.add_last(3)
lst.add_last(5)
lst.add_last(7)
lst.add_last(9)
lst.printlist()
lst.head.next = reverseBetween(lst.head.next, 2, 4)
lst.printlist()
lst.head.next = reverseBetween(lst.head.next, 1, 4)
lst.printlist()
lst.head.next = reverseBetween(lst.head.next, 1, 3)
lst.printlist()
