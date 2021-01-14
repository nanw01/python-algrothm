
from LinkedList import Node, LinkedList


def swapPairs(head):

    dummy = curr = Node(0)
    dummy.next = head

    while curr.next and curr.next.next:
        p1 = curr.next
        p2 = curr.next.next

        curr.next = p2
        p1.next = p2.next
        p2.next = p1
        curr = curr.next.next
    return dummy.next


lst = LinkedList()
lst.add_last(1)
lst.add_last(2)
lst.add_last(3)
lst.add_last(4)
lst.add_last(1)
lst.add_last(2)
lst.add_last(3)
lst.add_last(4)
lst.printlist()
lst.head.next = swapPairs(lst.head.next)
lst.printlist()
lst.head.next = swapPairs(lst.head.next)
lst.printlist()
