# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def merge(self, arrayOfArrays):
        """
        :type arrayOfArrays: List[ListNode]
        :rtype: ListNode
        """
        from heapq import *
        pq = []
        for i in range(len(arrayOfArrays)):
            if arrayOfArrays[i]:
                heappush(pq, (arrayOfArrays[i].val, i))
                arrayOfArrays[i] = arrayOfArrays[i].next

        dummy = ListNode(1)
        p = dummy
        while pq:
            val, idx = heappop(pq)
            p.next = ListNode(val)
            p = p.next
            if arrayOfArrays[idx]:
                heappush(pq, (arrayOfArrays[idx].val, idx))
                arrayOfArrays[idx] = arrayOfArrays[idx].next
        return dummy.next
