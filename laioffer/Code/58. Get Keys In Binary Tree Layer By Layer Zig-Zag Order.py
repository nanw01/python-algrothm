# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def zigZag(self, root):
        """
        input: TreeNode root
        return: Integer[]
        """
        # write your solution here
        from collections import deque
        if root is None:
            return root
        reverse = True
        res = []
        q = [root]
        next = []
        line = []
        while q:
            head = q.pop(0)
            if head.left:
                next.append(head.left)
            if head.right:
                next.append(head.right)
            line.append(head.val)
            if not q:
                if reverse:
                    res.extend(line[::-1])
                else:
                    res.extend(line)
                if next:
                    q = next
                    reverse = not reverse
                    next = []
                    line = []

        return res
