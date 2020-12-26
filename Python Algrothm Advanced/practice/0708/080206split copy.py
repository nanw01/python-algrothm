# 1.0.6  Ex.6 Split in Half
# Give a list, split in into two lists, one for the front half, and one for the back half.

# 对半分，使用双指针技巧

from LinkedList import Node, LinkedList


def split(head):
    if head is None:
        return

    fast = slow = front_end = head
    while fast is not None:
        front_end = slow
        slow = slow.next
        fast = fast.next.next if fast.next is not None else None
    
    front_end.next = None

    return (head,slow)




node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(4)
node5 = Node(5)
node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node5


# node6 = Node(6)
# node5.next = node6

front_node = Node()
back_node = Node()

front_node, back_node = split(node1)
front = LinkedList()
front.head.next = front_node
front.printlist()

back = LinkedList()
back.head.next = back_node
back.printlist()
