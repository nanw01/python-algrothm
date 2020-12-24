def deleteDuplicates(head):
    if head == None:
        return head

    node = head

    while node.next:
        if node.value == node.next.value:
            node.next = node.next.next
        else:
            node = node.next

    return head



lst = LinkedList()
lst.add_last(1)
lst.add_last(3)
lst.add_last(3)
lst.add_last(3)
lst.add_last(5)
lst.add_last(7)
lst.add_last(7)
lst.add_last(9)
lst.head.next = deleteDuplicates(lst.head.next)
lst.printlist()