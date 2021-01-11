# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def reverseInPairs(self, head):
        """
        input: ListNode head
        return: ListNode
        """
        # write your solution here
        dummy = cur = ListNode(0)
        dummy.next = head

        while cur.next and cur.next.next:
            p1 = cur.next
            p2 = cur.next.next
            cur.next = p2
            p1.next = p2.next
            p2.next = p1
            cur = cur.next.next

        return dummy.next

    def swapPairs(self, head):
        dummy = cur = ListNode(0)
        dummy.next = head

        while cur.next and cur.next.next:
            p1 = cur.next
            p2 = cur.next.next
            cur.next = p2
            p1.next = p2.next
            p2.next = p1
            cur = cur.next.next
        return dummy.next


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
node = s.reverseInPairs(n0)

while node:
    print(node.val, end=' ')
    node = node.next
