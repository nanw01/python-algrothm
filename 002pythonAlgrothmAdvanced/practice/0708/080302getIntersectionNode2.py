# 1.0.2  Ex.2 Intersection of Two Linked Lists
# Write a program to find the node at which the intersection of two singly linked lists begins.

# For example, the following two linked lists:

# A: a1 → a2

#                ↘

#                  c1 → c2 → c3

#                ↗    
# B: b1 → b2 → b3

# begin to intersect at node c1.


def getIntersectionNode2(headA, headB):
    if headA and headB:
        A, B = headA, headB
        while A!=B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A