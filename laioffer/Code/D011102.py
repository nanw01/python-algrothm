class ListNode():
    def __init__(self, value, next=None):
        self.val = value
        self.next = next


def removevowel(head):
    if not head:
        return None

    dummy = prev = ListNode(0)
    prev.next = curr = head
    while curr:
        if curr.val in ['a', 'e', 'i', 'o', 'u']:
            prev.next = curr.next
        else:
            prev = curr
        curr = curr.next
    return dummy.next


def printList(node):
    while node is not None:
        print(node.val, end=' ')
        node = node.next
    print()


n4 = ListNode('e')
n3 = ListNode('d', n4)
n2 = ListNode('c', n3)
n1 = ListNode('b', n2)
n0 = ListNode('a', n1)
printList(n0)
# aa = removevowel(n0)
# printList(aa)


def reverse(head):

    prev = None
    while head:
        nxtNode = head.next
        head.next = prev
        prev = head
        head = nxtNode

    return prev


aa = reverse(n0)
printList(aa)


def reverseList(head):

    if head == None or head.next == None:
        return head
    p = reverseList(head.next)
    head.next.next, head.next = head, None
    return p


aa = reverseList(aa)
printList(aa)


def lenth(node):
    l = 0
    while node:
        l += 1
        node = node.next
    return l


def add_list(node1, node2):

    n1 = reverseList(node1)
    n2 = reverseList(node2)

    fake_head = cur_node = ListNode(None)
    carry = 0
    while n1 and n2:

        tem_sum = n1.val + n2.val + carry
        carry = tem_sum // 10
        cur_node.next = ListNode(tem_sum % 10)
        cur_node = cur_node.next
        n1, n2 = n1.next, n2.next
    while n1:
        tem_sum = n1.val + carry
        carry = tem_sum // 10
        cur_node.next = ListNode(tem_sum % 10)
        cur_node = cur_node.next
        n1 = n1.next
    while n2:
        tem_sum = n2.val + carry
        carry = tem_sum // 10
        cur_node.next = ListNode(tem_sum % 10)
        cur_node = cur_node.next
        n2 = n2.next
    if carry > 0:
        cur_node.next = ListNode(carry)

    return reverse(fake_head.next)


n10 = ListNode(9)
n11 = ListNode(6, n10)
n12 = ListNode(5, n11)

n20 = ListNode(1)
n21 = ListNode(3, n20)
n22 = ListNode(4, n21)

printList(n12)
printList(n22)

n12 = add_list(n12, n22)
printList(n12)
