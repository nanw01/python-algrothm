
from LinkedList import Node, LinkedList


def swapPairs(head):

    dummy=cur=Node(0)

    dummy.next = head

    while cur.next and cur.next.next:
        p1 = cur.next
        p2=cur.next.next

        cur.next = p2
        p1.next = p2.next
        p2.next=p1
        cur=cur.next.next

    return dummy.next


lst = LinkedList()
lst.add_last(1)
lst.add_last(2)
lst.add_last(3)
lst.add_last(4)
lst.add_last(5)
lst.add_last(6)
lst.add_last(7)
lst.add_last(8)
lst.printlist()
lst.head.next = swapPairs(lst.head.next)
lst.printlist()
lst.head.next = swapPairs(lst.head.next)
lst.printlist()
