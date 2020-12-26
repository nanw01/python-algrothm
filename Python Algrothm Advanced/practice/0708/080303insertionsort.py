# 1.0.3  Ex.3 Insertion Sort List

from LinkedList import Node,LinkedList

def insertionSortList(head):
    dummy = Node(0)
    cur = head

    while cur is not None:

        pre = cur
        while pre is not None and pre.value <



def findmin(node):

    cur = node
    while cur is not None and cur.next is not None:
        if cur.value < cur.next.value:

            


node1 = Node(-9)
node2 = Node(1)
node3 = Node(-13)
node4 = Node(6)
node5 = Node(5)

node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node5
lst = LinkedList()
lst.head.next = node1
lst.printlist()

node = insertionSortList(node1)

lst.head.next = node
lst.printlist()