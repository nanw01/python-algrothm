from LinkedList import LinkedList

# 翻转单链表


def reverse(lst):
    head = lst.head
    curr = head.next
    pre = None
    nxt = None

    while curr:
        nxt = curr.next

        curr.next = pre
        pre = curr
        curr = nxt

    head.next = pre
    

lst = LinkedList()
lst.add_last(1)
lst.add_last(3)
lst.add_last(5)
lst.add_last(7)
lst.add_last(9)
lst.printlist()
reverse(lst)
lst.printlist()
