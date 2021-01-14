from LinkedList import LinkedList
from LinkedList import Node

def deleteDuplicates2(head):
    dummy = pre = Node(0)
    dummy.next = head
    
    while head and head.next:
        if head.value == head.next.value:
            while head and head.next and head.value == head.next.value:
                head = head.next
            head = head.next
            pre.next = head
        else:
            head = head.next
            pre = pre.next

    return dummy.next
            



lst = LinkedList()
lst.add_last(1)
lst.add_last(3)
lst.add_last(3)
lst.add_last(3)
lst.add_last(5)
lst.add_last(7)
lst.add_last(7)
lst.add_last(9)
lst.head.next = deleteDuplicates2(lst.head.next)
lst.printlist()