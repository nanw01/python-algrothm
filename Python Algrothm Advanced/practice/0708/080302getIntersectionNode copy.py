# 1.0.2  Ex.2 Intersection of Two Linked Lists
# Write a program to find the node at which the intersection of two singly linked lists begins.

# For example, the following two linked lists:

# A: a1 → a2

#                ↘

#                  c1 → c2 → c3

#                ↗
# B: b1 → b2 → b3

# begin to intersect at node c1.

from LinkedList import Node, LinkedList


def getIntersectionNode(headA, headB):

    lenA = lenB = 0
    tempA, tempB = headA, headB

    while tempA is not None:
        lenA += 1
        tempA = tempA.next

    while tempB is not None:
        lenB += 1
        tempB = tempB.next

    tempA, tempB = headA, headB
    if lenA > lenB:
        for _ in range(lenA-lenB):
            tempA = tempA.next

    else:
        for _ in range(lenB-lenA):
            tempB = tempB.next

    while tempA is not tempB:
        tempA = tempA.next
        tempB = tempB.next

    return tempA



a1 = Node('a1')
a2 = Node('a2')

b1 = Node('b1')
b2 = Node('b2')
b3 = Node('b3')

c1 = Node('c1')
c2 = Node('c2')
c3 = Node('c3')


a1.next = a2
a2.next = c1

b1.next = b2
b2.next = b3
b3.next = c1

c1.next = c2
c2.next = c3

print(getIntersectionNode(a1, b1).value)
