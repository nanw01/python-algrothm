# 是否有环
from LinkedList import Node,LinkedList

def has_cycle_helper(node):

    if node is None:
        return False

    head = node
    fast=slow=head
    while fast is not None and fast.next is not None:
        fast = fast.next.next
        slow = slow.next
        if fast ==slow:
            return True

    return False



node1 = Node(1)
print(has_cycle_helper(node1))
node2 = Node(2)
node3 = Node(3)
node1.next = node2
node2.next = node3
print(has_cycle_helper(node1))
node3.next = node1
print(has_cycle_helper(node1))