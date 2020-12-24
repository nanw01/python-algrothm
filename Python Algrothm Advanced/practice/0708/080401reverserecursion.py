from LinkedList import LinkedList

# 翻转单链表
def reverseRecursion(node):
    if (node is None or node.next is None):
        return node
    p = reverseRecursion(node.next)
    node.next.next = node
    node.next = None
    return p


    
lst = LinkedList()
lst.add_last(1)
lst.add_last(3)
lst.add_last(5)
lst.add_last(7)
lst.add_last(9)
lst.printlist()
lst.head.next = reverseRecursion(lst.head.next)
lst.printlist()