# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def maxPathSum(self, root):
        """
        input: TreeNode root
        return: int
        """
        # write your solution here
        self.max = -float('inf')
        self.helper(root)
        return self.max

    def helper(self, node):
        if node is None:
            return
        l = self.helper(node.left)
        r = self.helper(node.right)
        self.max = max(self.max, node.val, node.val+l, node.val+r)

        return max(node.val, node.val+l, node.val + r)
