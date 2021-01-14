from LinkedList import LinkedList

# 翻转单链表
def reverse(lst):

    head = lst.head
    result =None
    curr = head.next
    nxt = None

    while curr is not None:
        nxt = curr.next
        curr.next = result
        result = curr
        curr = nxt

    head.next = result

lst = LinkedList()
lst.add_last(1)
lst.add_last(3)
lst.add_last(5)
lst.add_last(7)
lst.add_last(9)
lst.printlist()
reverse(lst)
lst.printlist()
