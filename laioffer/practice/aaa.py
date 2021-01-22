class ListNode:

    def __init__(self, x):
        self.val = x
        self.next = None


def remove_node(lst, target):
    if not lst:
        return lst

    dummy = curr = ListNode(None)
    curr.next = lst

    while curr:
        if curr.next.val == target:
            curr.next = curr.next.next
        curr = curr.next

    return dummy.next


l1 = ListNode(10)
l2 = ListNode(8)
l3 = ListNode(8)
l4 = ListNode(1)
l5 = ListNode(8)

l1.next = l2
l2.next = l3
l3.next = l4
l4.next = l5

node = remove_node(l1, 8)

while node:
    print(node.val, end=' ')
    node = node.next
