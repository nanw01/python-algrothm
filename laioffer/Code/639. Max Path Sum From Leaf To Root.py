# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def maxPathSumLeafToRoot(self, root):
        """
        input: TreeNode root
        return: int
        """
        # write your solution here
        self._max_sum = float('-inf')
        self._maxPathSumLeafToRoot(root, 0)
        return self._max_sum

    def _maxPathSumLeafToRoot(self, node, cur):
        if node is None:
            return float('-inf')

        if node.left is None and node.right is None:
            self._max_sum = max(self._max_sum, cur + node.val)
            return

        self._maxPathSumLeafToRoot(node.left, cur + node.val)
        self._maxPathSumLeafToRoot(node.right, cur + node.val)

        return
