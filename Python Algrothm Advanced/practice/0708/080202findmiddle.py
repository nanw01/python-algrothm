# 1.0.2  Ex.2 Find the Middle Node

from LinkedList import Node, LinkedList


def find_middle(list):

    a = b = list.head
    while a is not None and a.next is not None:
        a = a.next.next
        b = b.next

    return b.value


lst = LinkedList()
# find_middle(lst)
lst.add_last(1)
print(find_middle(lst))
lst.printlist()
lst.add_last(2)
lst.add_last(3)
lst.add_last(4)
print(find_middle(lst))
lst.printlist()
lst.add_last(5)
print(find_middle(lst))
lst.printlist()
