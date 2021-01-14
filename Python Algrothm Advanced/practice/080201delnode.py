# 1.0.1  Ex.1 Delete Node
# Delete Node in Linked List: except the tail, given only access to that node.

from LinkedList import Node, LinkedList


def delete_node(node):
    if node.next:
        node.value = node.next.value
        node.next = node.next.next


lst = LinkedList()
lst.add_last(1)
lst.add_last(2)
lst.add_last(3)
lst.add_last(4)
lst.printlist()
delete_node(lst.head.next.next)
lst.printlist()
