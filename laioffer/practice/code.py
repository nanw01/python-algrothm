# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        return self._lowestCommonAncestor(root, p, q)

    def _lowestCommonAncestor(self, node, p, q):
        if node is None:
            return node

        if node == q or node == p:
            return node

        left = self._lowestCommonAncestor(node.left, p, q)
        right = self._lowestCommonAncestor(node.right, p, q)

        if left and right:
            return node
        if left:
            return left
        return right
