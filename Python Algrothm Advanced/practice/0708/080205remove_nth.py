# 1.0.5  Ex.5 Remove Nth to Last
# Remove the nth to last element of a singly linked list
# 两个 pointers


from LinkedList import Node, LinkedList


def remove_nth(lst, n):

    assert n > 0 and n <= lst.length

    fast = lst.head
    while n > 0:
        fast = fast.next
        n -= 1

    slow = lst.head

    while fast is not None and fast.next is not None:
        fast = fast.next
        slow = slow.next

    result = slow.next
    slow.next = slow.next.next

    lst.length -= 1

    return result


lst = LinkedList()
lst.add_last(1)
lst.add_last(3)
lst.add_last(5)
lst.add_last(7)
lst.add_last(9)

lst.printlist()
print(remove_nth(lst, 3).value)
lst.printlist()
