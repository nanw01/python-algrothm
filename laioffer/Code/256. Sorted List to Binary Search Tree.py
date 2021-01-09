# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

# Definition for a binary tree node.


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def sortedListToBST(self, head):
        """
        input: ListNode head
        return: TreeNode
        """
        # write your solution here
        arr = []
        while head:
            arr.append(head.val)
            head = head.next
        return self._createBST(arr)

    def _createBST(self, arr):
        if not arr:
            return None
        return self._bst(arr, 0, len(arr)-1)

    def _bst(self, arr, start, end):
        if start > end:
            return None
        mid = (start + end) // 2
        root = TreeNode(arr[mid])
        root.left = self._bst(arr, start, mid-1)
        root.right = self._bst(arr, mid + 1, end)
        return root

    def print_inOrder(self, root):
        if not root:
            return []
        return self.print_inOrder(root.left)+[root.val]+self.print_inOrder(root.right)


lst = [16, 36, 39, 42, 76, 82, 86, 107, 117, 149, 153, 186, 188, 193, 204, 217, 237, 246, 248, 250, 255, 256, 257, 258, 259, 280, 300, 320, 321, 341, 350, 354, 364, 367, 371, 376, 385, 410, 413, 465, 474, 506, 511, 536,
       549, 564, 568, 590, 605, 609, 620, 630, 639, 658, 668, 678, 683, 685, 700, 702, 714, 733, 752, 762, 779, 786, 810, 811, 812, 822, 825, 847, 854, 869, 874, 886, 896, 928, 930, 943, 945, 958, 963, 990, 992, 993]

lstNode = [ListNode(lst[i]) for i in range(len(lst))]

for i in range(len(lstNode)-1):
    lstNode[i].next = lstNode[i + 1]

s = Solution()

print(s.print_inOrder(s.sortedListToBST(lstNode[0])))
