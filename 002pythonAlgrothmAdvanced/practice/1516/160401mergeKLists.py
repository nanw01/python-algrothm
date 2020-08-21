from queue import PriorityQueue
from LinkedList import LinkedList
from LinkedList import Node


def mergeKLists(lists):
    dummy = Node(None)
    curr = dummy
    q = PriorityQueue()
    for node in lists:
        if node:
            q.put((node.value, node))
    while q.qsize() > 0:
        curr.next = q.get()[1]
        curr = curr.next
        if curr.next:
            q.put((curr.next.value, curr.next))
    return dummy.next


lst1 = LinkedList()
lst1.add_last(1)
lst1.add_last(4)
lst1.add_last(5)

lst2 = LinkedList()
lst2.add_last(1)
lst2.add_last(3)
lst2.add_last(4)

lst3 = LinkedList()
lst3.add_last(2)
lst3.add_last(6)

lists = [lst1.head.next, lst2.head.next, lst3.head.next]
node = mergeKLists(lists)
result = LinkedList()

result.head.next = node
result.printlist()
