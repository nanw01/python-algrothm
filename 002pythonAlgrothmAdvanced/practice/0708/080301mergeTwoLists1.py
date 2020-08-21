from LinkedList import Node
from LinkedList import LinkedList

# iteratively
# O(m + n)


def mergeTwoLists1(l1, l2):
    dummy = cur = Node(0)
    while l1 and l2:
        if l1.value < l2.value:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next


node11 = Node(1)
node12 = Node(2)
node14 = Node(4)
node11.next = node12
node12.next = node14

node21 = Node(1)
node23 = Node(3)
node24 = Node(4)
node21.next = node23
node23.next = node24


node = mergeTwoLists1(node11, node21)
lst = LinkedList()
lst.head.next = node
lst.printlist()
