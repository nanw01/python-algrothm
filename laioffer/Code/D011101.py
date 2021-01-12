class ListNode():
    def __init__(self, value, next=None):
        self.val = value
        self.nxt = next


def removevowel(head):
    if not head:
        return None

    dummy = prev = ListNode(0)
    prev.next = curr = head
    while curr:
        if curr.val in ['a', 'e', 'i', 'o', 'u']:
            prev.nxt = curr.nxt
        else:
            prev = curr
        curr = curr.nxt
    return dummy.nxt


def printList(node):
    while node is not None:
        print(node.val, end=' ')
        node = node.nxt
    print()


n4 = ListNode('c')
n3 = ListNode('e', n4)
n2 = ListNode('m', n3)
n1 = ListNode('i', n2)
n0 = ListNode('a', n1)
printList(n0)
aa = removevowel(n0)
printList(aa)
