# 1.0.2  Ex.2 Intersection of Two Linked Lists
# Write a program to find the node at which the intersection of two singly linked lists begins.

# For example, the following two linked lists:

# A: a1 → a2

#                ↘

#                  c1 → c2 → c3

#                ↗    
# B: b1 → b2 → b3

# begin to intersect at node c1.


def getIntersectionNode(headA, headB):
    curA, curB = headA, headB
    lenA, lenB = 0, 0
    while curA is not None:
        lenA += 1
        curA = curA.next
    while curB is not None:
        lenB += 1
        curB = curB.next
    curA, curB = headA, headB
    if lenA > lenB:
        for _ in range(lenA-lenB):
            curA = curA.next
    elif lenB > lenA:
        for _ in range(lenB-lenA):
            curB = curB.next
    while curB != curA:
        curB = curB.next
        curA = curA.next
    return curA