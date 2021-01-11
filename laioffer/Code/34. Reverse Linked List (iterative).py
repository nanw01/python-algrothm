# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def reverse(self, head):
        """
        input: ListNode head
        return: ListNode
        """
        # write your solution here

        prev = None
        current = head
        while(current is not None):
            next = current.next
            current.next = prev
            prev = current
            current = next
        return prev


n0 = ListNode(0)
n1 = ListNode(1)
n2 = ListNode(2)
n3 = ListNode(3)
n4 = ListNode(4)
n5 = ListNode(5)

n0.next = n1
n1.next = n2
n2.next = n3
n3.next = n4
n4.next = n5


s = Solution()
node = n0
while node:
    print(node.val, end=' ')
    node = node.next
print()
node = s.reverse(n0)

while node:
    print(node.val, end=' ')
    node = node.next
