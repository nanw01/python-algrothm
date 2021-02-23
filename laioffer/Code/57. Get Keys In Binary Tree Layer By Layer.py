# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def layerByLayer(self, root):
        """
        input: TreeNode root
        return: int[][]
        """
        # write your solution here
        if root is None:
            return []
        ret = []
        q = [root]
        next = []
        line = []
        while q:
            node = q.pop(0)
            if node.left:
                next.append(node.left)
            if node.right:
                next.append(node.right)
            line.append(node.val)

            if not q:
                ret.append(line)
                if next:
                    q = next
                    next = []
                    line = []

        return ret
