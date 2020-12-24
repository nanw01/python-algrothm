# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
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
    head = prev

    