# 1.0.2  Ex.2 Intersection of Two Linked Lists
# Write a program to find the node at which the intersection of two singly linked lists begins.

# For example, the following two linked lists:

# A: a1 → a2

#                ↘

#                  c1 → c2 → c3

#                ↗    
# B: b1 → b2 → b3

# begin to intersect at node c1.


from LinkedList import Node,LinkedList

def getIntersectionNode2(headA, headB):
    if headA and headB:
        a,b = headA ,headB


        while a is not b:
            a = a.next if a else headB
            b = b.next if b else headA
            
        return a


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

print(getIntersectionNode2(a1, b1).value)