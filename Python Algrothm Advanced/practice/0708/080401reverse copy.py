from LinkedList import LinkedList

# 翻转单链表


def reverse(lst):
    pre = None
    head= lst.head
    curr= head.next
    nxt = None
    while curr is not None:
        nxt = curr.next
        curr.next = pre
        
        pre = curr
        curr=  nxt
    
    lst.head.next = pre


lst = LinkedList()
lst.add_last(1)
lst.add_last(3)
lst.add_last(5)
lst.add_last(7)
lst.add_last(9)
lst.printlist()
reverse(lst)
lst.printlist()
